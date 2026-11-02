# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Ascend fused RMSNorm for the transformer stack.

``torch.nn.RMSNorm`` decomposes into five separate AIV kernels on Ascend
(``Pows``, ``ReduceMean``, ``Muls``, ``Rsqrt``, ``Mul``).  ``torch_npu.npu_rms_norm``
does the same arithmetic in one kernel and is 7.3x faster on the production 480p
shape (2.62 ms -> 0.36 ms at 15730 rows x 5376 channels).

**Off by default.**  The fused op matches ATen to 1e-9 in a standalone comparison
and to 0.2 % relative mean inside a real transformer block -- i.e. below one bf16
ulp -- but six-step consistency sampling amplifies a per-block ulp into a
visible change of the final video at the same seed (measured: 20 % mean absolute
pixel difference, frame correlation 0.90).  The released contract is bit-exact
output, so this is opt-in via ``PIKA_FUSED_RMSNORM=1``: -5.5 % denoising wall
time and +6 pp cube utilisation, at the cost of no longer being bit-identical.

The fusion is applied **per module instance** to the transformer only.  Applying
it process-wide would also change the Qwen3-VL text encoder and the VAEs, which
perturbs the conditioning embedding before the sampler ever runs.
"""
from __future__ import annotations

import torch

from . import accelerator

#: Relative-mean tolerance used by the numerical A/B check.
FUSION_TOLERANCE = 1e-6


class FusedRMSNorm(torch.nn.Module):
    """Weight-only RMSNorm backed by the Ascend fused operator.

    Mirrors ``nn.RMSNorm`` for the two shapes the transformer uses: a full
    hidden-size norm and a per-head norm.  Non-affine instances fall back to the
    ATen path, which is what the decoder-side norms rely on.
    """

    def __init__(self, normalized_shape, eps: float, weight: torch.Tensor | None):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(int(v) for v in normalized_shape)
        self.eps = float(eps)
        self.weight = weight
        import torch_npu  # registers the fused operators; import is idempotent

        self._fused = torch_npu.npu_rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if w is None:
            return torch.nn.functional.rms_norm(x, self.normalized_shape, None, self.eps)
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        y, _ = self._fused(x, w, self.eps)
        return y

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, fused=True"


def _convert(module: torch.nn.Module) -> int:
    swapped = 0
    for name, child in list(module.named_children()):
        if type(child) is torch.nn.RMSNorm:
            fused = FusedRMSNorm(child.normalized_shape, child.eps, child.weight)
            setattr(module, name, fused)
            swapped += 1
        else:
            swapped += _convert(child)
    return swapped


def fuse_attention_norms(model: torch.nn.Module) -> int:
    """Swap the per-head QK norms for the fused Ascend kernel.

    Unlike :func:`fuse_rmsnorm_modules` this is **bit-exact** and therefore
    always on.  The QK norms normalise over ``head_dim`` (128 channels), and on
    that width ``npu_rms_norm`` reproduces ATen to the bit: verified over 24
    random bf16 tensors and on the real Q/K of a forward pass (``torch.equal``
    true in every case, max |diff| 0).

    Why the narrower norm can be fused but the full-width one cannot: ATen
    decomposes RMSNorm into five AIV kernels (``Pows``, ``ReduceMean``,
    ``Muls``, ``Rsqrt``, ``Mul``), and on 128 channels those launches dominate.
    ``npu_rms_norm`` does the same arithmetic in one kernel, 24x faster
    (23.4 ms -> 0.96 ms per call at 37643 rows x 56 heads x 128).  The
    full-width 5376-channel norm is a different case: there the two
    implementations accumulate in a different order and differ by up to one
    bf16 ulp, which six-step sampling amplifies -- see
    :func:`fuse_rmsnorm_modules`.

    Only modules that own a ``norm_q``/``norm_k`` pair whose ``normalized_shape``
    matches ``head_dim`` are touched, so this cannot reach the VAE or the text
    encoder.  Returns the number of swapped modules (0 off NPU).
    """
    if accelerator.backend() != "npu":
        return 0
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return 0

    swapped = 0
    for module in model.modules():
        head_dim = getattr(module, "head_dim", None)
        if not head_dim:
            continue
        for attr in ("norm_q", "norm_k"):
            norm = getattr(module, attr, None)
            if type(norm) is not torch.nn.RMSNorm:
                continue
            shape = tuple(norm.normalized_shape)
            if shape != (int(head_dim),):
                continue
            setattr(module, attr, FusedRMSNorm(shape, norm.eps, norm.weight))
            swapped += 1
    return swapped


def fuse_rmsnorm_modules(model: torch.nn.Module) -> int:
    """Replace every ``nn.RMSNorm`` under ``model`` with :class:`FusedRMSNorm`.

    Opt-in (``PIKA_FUSED_RMSNORM=1``) because the full-width hidden-size norm is
    *not* bit-exact: see the module docstring.  Use
    :func:`fuse_attention_norms` for the exact part.

    Returns the number of swapped modules (0 when the backend cannot fuse).
    """
    if accelerator.backend() != "npu":
        return 0
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return 0
    return _convert(model)
