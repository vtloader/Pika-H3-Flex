# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Backend-agnostic accelerator access for CUDA and Ascend NPU.

The runtime selects one backend explicitly and routes every device-specific
call through this module. Selection order:

1. ``PIKA_DEVICE`` (``cuda`` | ``npu`` | ``cpu``, case-insensitive), if set;
2. Ascend NPU, when ``torch_npu`` imports and reports an available device;
3. CUDA, when ``torch.cuda.is_available()``;
4. CPU.

``torch_npu`` is imported lazily and caught when missing, so a CUDA-only
installation never needs it. On CUDA every helper keeps the previous
``torch.cuda`` semantics, including CUDA streams and the TF32 flags.
"""

from __future__ import annotations

import functools
import inspect
import os

import torch

DEVICE_ENV = "PIKA_DEVICE"
BACKENDS = ("cuda", "npu", "cpu")


def _import_torch_npu():
    """Import torch_npu (registering the NPU backend) without requiring it."""
    try:
        import torch_npu  # noqa: F401
    except Exception:  # torch_npu missing, or no CANN/NPU driver
        return None
    return getattr(torch, "npu", None)


def _available_npu():
    """``torch.npu`` when an Ascend device is usable, else ``None``."""
    npu = _import_torch_npu()
    if npu is None:
        return None
    try:
        return npu if npu.is_available() else None
    except Exception:  # driver/library mismatch: treat as unavailable
        return None


@functools.lru_cache(maxsize=1)
def backend() -> str:
    """Selected backend: ``"npu"``, ``"cuda"`` or ``"cpu"`` (resolved once)."""
    requested = os.environ.get(DEVICE_ENV, "").strip().lower()
    if requested:
        if requested not in BACKENDS:
            raise ValueError(f"{DEVICE_ENV} must be one of {BACKENDS}, got {requested!r}")
        if requested == "npu" and _available_npu() is None:
            raise RuntimeError(f"{DEVICE_ENV}=npu but torch_npu/NPU is not available")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{DEVICE_ENV}=cuda but CUDA is not available")
        return requested
    if _available_npu() is not None:
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _module():
    """``torch.cuda`` / ``torch.npu`` for the selected backend, else ``None``."""
    kind = backend()
    if kind == "cuda":
        return torch.cuda
    if kind == "npu":
        module = getattr(torch, "npu", None)
        return module if module is not None else _import_torch_npu()
    return None


def reset_backend_cache() -> None:
    """Resolve the backend again on the next call (tests only; selection is cached)."""
    backend.cache_clear()
    _fork_rng_has_device_type.cache_clear()


def is_available() -> bool:
    """Whether the selected backend is an available accelerator (CPU returns False)."""
    module = _module()
    return False if module is None else bool(module.is_available())


def device_count() -> int:
    module = _module()
    return 0 if module is None else int(module.device_count())


def current_device() -> int:
    module = _module()
    return 0 if module is None else int(module.current_device())


def device_name() -> str:
    module = _module()
    if module is None:
        return "cpu"
    return str(module.get_device_name(current_device()))


def device(index: int | None = None) -> torch.device:
    kind = backend()
    if kind == "cpu":
        return torch.device("cpu")
    return torch.device(kind if index is None else f"{kind}:{int(index)}")


def empty_cache() -> None:
    module = _module()
    if module is not None:
        module.empty_cache()


def synchronize() -> None:
    module = _module()
    if module is not None:
        module.synchronize()


def reset_peak_memory_stats() -> None:
    module = _module()
    if module is not None:
        module.reset_peak_memory_stats()


def max_memory_allocated() -> int:
    module = _module()
    return 0 if module is None else int(module.max_memory_allocated())


def Stream(device=None, **kwargs):  # noqa: N802 - mirrors torch.cuda.Stream
    """Backend stream factory; ``None`` on CPU (streams are an accelerator concept)."""
    module = _module()
    if module is None:
        return None
    if device is not None:
        kwargs["device"] = device
    return module.Stream(**kwargs)


def supports_streams() -> bool:
    """Whether streamed group offload is usable on the selected backend.

    diffusers-only detection covers CUDA and XPU; Ascend NPU has an equivalent
    primary/side stream API (``torch.npu.Stream``, ``wait_stream``,
    ``synchronize``) that the same prefetch pipeline can drive, so it is handled
    by :func:`enable_group_offload` rather than by the stock entry point.
    """
    return backend() in ("cuda", "npu")


def _apply_group_offload_direct(module, *, onload_device, offload_device,
                                num_blocks_per_group, record_stream, low_cpu_mem_usage,
                                stream):
    """Apply diffusers' group offload with an explicit transfer stream.

    ``diffusers.hooks.apply_group_offloading`` builds the stream only for CUDA
    and XPU and raises for anything else, even though the rest of the machinery
    is backend-agnostic: ``ModuleGroup`` resolves the accelerator module from
    ``torch.accelerator.current_accelerator().type`` (``torch.npu`` here) and
    only uses ``stream(...)``, ``current_stream()`` and ``synchronize()``. Build
    the config ourselves and hand it to the same private entry point so the
    prefetch pipeline runs unchanged.
    """
    from diffusers.hooks import group_offloading as go

    config = go.GroupOffloadingConfig(
        onload_device=torch.device(onload_device),
        offload_device=torch.device(offload_device),
        offload_type=go.GroupOffloadingType.BLOCK_LEVEL,
        non_blocking=True,
        record_stream=bool(record_stream),
        low_cpu_mem_usage=bool(low_cpu_mem_usage),
        num_blocks_per_group=int(num_blocks_per_group),
        stream=stream,
    )
    go._apply_group_offloading(module, config)


def mem_get_info(index: int | None = None) -> tuple[int, int]:
    """``(free, total)`` bytes on the selected accelerator; ``(0, 0)`` when unknown."""
    module = _module()
    if module is None or not hasattr(module, "mem_get_info"):
        return 0, 0
    try:
        free, total = module.mem_get_info() if index is None else module.mem_get_info(int(index))
        return int(free), int(total)
    except Exception:  # driver/CANN without the query, or a stale context
        return 0, 0


def memory_reserved() -> int:
    module = _module()
    if module is None or not hasattr(module, "memory_reserved"):
        return 0
    try:
        return int(module.memory_reserved())
    except Exception:
        return 0


def activation_bytes(sequence_length: int, hidden_size: int, *, dtype_bytes: int = 2,
                     copies: int = 15) -> int:
    """Peak activation footprint of one transformer forward.

    Calibrated from two measured TP2 points rather than guessed:

    * 480p/5 s (``sequence_length`` 15 730): 34.79 GiB allocated with 32.5 GiB of
      weights -> 2.29 GiB of activations;
    * 1920x1088/10 s (``sequence_length`` 148 580): OOM at 49.77 GiB allocated
      plus a 3.95 GiB request -> at least 21.2 GiB of activations.

    Both give ``activations / (seq * hidden * 2 bytes)`` of 14.3-14.5, so the
    coefficient is the number of full-width bf16 buffers that stay live at once
    (residual, normalised input, Q/K/V, attention out, SwiGLU halves, the fp32
    reduce buffer, attention workspace). 15 is the measured upper bound plus a
    small margin.
    """
    rows = int(sequence_length) * int(hidden_size) * int(dtype_bytes)
    return int(rows) * int(copies)


def fits_resident(module, *, onload_device, sequence_length: int, hidden_size: int,
                  headroom_frac: float = 0.10) -> tuple[bool, dict]:
    """Whether ``module``'s weights plus this request's activations fit in HBM.

    Judging by weights alone is not enough: a TP shard is 32.5 GiB and fits an
    idle 61 GiB chip, yet 1920x1088/10 s activations add ~22 GiB and the run OOMs
    mid-sampling.

    The budget uses ``min(free, total)`` and keeps ``headroom_frac`` on top:
    ``free`` already accounts for what PyTorch reserved earlier in the process
    (measured 58.28 GiB reserved when the 1920x1088/10 s run died), so judging
    against the *total* capacity alone over-estimates what is really available.
    """
    target = torch.device(device() if onload_device is None else onload_device)
    weights = sum(p.numel() * p.element_size() for p in module.parameters())
    free, total = mem_get_info(target.index)
    if not total:
        return False, {"reason": "no memory query", "weights": weights}
    act = activation_bytes(sequence_length, hidden_size)
    need = weights + act
    available = min(free, total) if free else total
    limit = int(available * (1.0 - float(headroom_frac)))
    info = {"weights_bytes": weights, "activation_bytes": act, "needed_bytes": need,
            "limit_bytes": limit, "total_bytes": total, "free_bytes": free}
    return need <= limit, info


def enable_group_offload(module, *, onload_device=None, offload_device=torch.device("cpu"),
                         num_blocks_per_group=1, use_stream=True, record_stream=False,
                         low_cpu_mem_usage=False, sequence_length: int | None = None,
                         hidden_size: int | None = None, headroom_frac: float = 0.10) -> bool:
    """Place transformer blocks, offloading only if weights **and activations** fit.

    The decision is **adaptive**, because whether a model fits depends on how it
    was split and on the request, not just on the model:

    * ``--tensor-parallel 2`` leaves ~32.5 GiB of weights per rank, which fits a
      61.27 GiB chip at 480p/5 s but not at 1920x1088/10 s -- activations add
      ~21 GiB and the run OOMs mid-sampling. So TP is not automatically resident
      either; ``sequence_length``/``hidden_size`` are required to judge it.
    * single-card keeps the full 61.7 GiB and never fits.

    When it fits, the module is simply moved to the device: same shards, same
    collectives, same kernels, so the output is bit-identical to the offloaded
    path (measured: identical mp4 md5 for TP2 at 480p/5 s).

    Otherwise, CUDA keeps diffusers' own streamed entry point. On NPU the blocking
    path costs a *synchronous, pageable* device-to-host copy after every block --
    measured at 11.5 GB/s, i.e. 50 % of the whole denoising phase. The streamed
    path instead keeps one pinned CPU copy per weight, so offloading is just
    re-pointing ``param.data`` (no D2H at all) and onloading happens on a side
    stream ahead of the block that needs it.

    ``low_cpu_mem_usage=False`` is deliberate: it makes the CPU copies pinned up
    front. Pinning on the fly costs ~1 s of extra host time per step and removes
    the async property that the overlap depends on.

    Returns True when blocks are streamed (offloaded), False when they stay resident.
    """
    target = torch.device(device() if onload_device is None else onload_device)
    if sequence_length is not None and hidden_size is not None:
        fits, info = fits_resident(module, onload_device=target,
            sequence_length=sequence_length, hidden_size=hidden_size,
            headroom_frac=headroom_frac)
        if fits:
            module.to(target)
            synchronize()
            return False
    else:
        # No request geometry: fall back to the weight-only test, which is only
        # safe when the caller has already bounded the activation footprint.
        weights = sum(p.numel() * p.element_size() for p in module.parameters())
        free, total = mem_get_info(target.index)
        if total and weights <= int(total * 0.85):
            module.to(target)
            synchronize()
            return False

    if not use_stream:
        module.enable_group_offload(onload_device=onload_device, offload_device=offload_device,
            offload_type="block_level", num_blocks_per_group=num_blocks_per_group,
            use_stream=False, record_stream=False, low_cpu_mem_usage=low_cpu_mem_usage)
        return True
    kind = backend()
    if kind == "npu":
        _apply_group_offload_direct(module, onload_device=target,
            offload_device=offload_device, num_blocks_per_group=num_blocks_per_group,
            record_stream=record_stream, low_cpu_mem_usage=low_cpu_mem_usage,
            stream=Stream(device=target))
        return True
    module.enable_group_offload(onload_device=onload_device, offload_device=offload_device,
        offload_type="block_level", num_blocks_per_group=num_blocks_per_group,
        use_stream=True, record_stream=record_stream, low_cpu_mem_usage=low_cpu_mem_usage)
    return True


def manual_seed(seed: int, target_device=None) -> None:
    """Seed the selected backend's generator, optionally for one device index (no-op on CPU)."""
    target = torch.device(target_device) if target_device is not None else None
    kind = target.type if target is not None else backend()
    if kind not in ("cuda", "npu"):
        return
    module = getattr(torch, kind, None)
    if module is None and kind == "npu":
        module = _import_torch_npu()
    if module is None:
        return
    index = target.index if target is not None else None
    if index is None:
        module.manual_seed(int(seed))
        return
    with module.device(index):
        module.manual_seed(int(seed))


def rng_devices(target_device=None) -> list[int]:
    """Device indices that ``torch.random.fork_rng`` must save for ``target_device``."""
    target = torch.device(target_device) if target_device is not None else device()
    if target.type not in ("cuda", "npu"):
        return []
    module = getattr(torch, target.type, None)
    if module is None and target.type == "npu":
        module = _import_torch_npu()
    if module is None:
        raise RuntimeError(f"device {target} is not available on this installation")
    if target.index is not None:
        return [int(target.index)]
    return [int(module.current_device())]


@functools.lru_cache(maxsize=1)
def _fork_rng_has_device_type() -> bool:
    return "device_type" in inspect.signature(torch.random.fork_rng).parameters


def fork_rng_kwargs(target_device=None) -> dict:
    """Keyword arguments for ``torch.random.fork_rng`` isolating CPU + ``target_device``."""
    target = torch.device(target_device) if target_device is not None else device()
    kwargs = {"devices": rng_devices(target)}
    if target.type in ("cuda", "npu") and _fork_rng_has_device_type():
        kwargs["device_type"] = target.type
    return kwargs


def offload_leaves(module, target_device=None) -> None:
    """Keep ``module`` on CPU and stream it onto the accelerator one leaf at a time.

    Leaf-level offloading is the only granularity that shrinks a single resident
    component below its full weight size, which the 62 GiB Qwen3-VL encoder needs.
    Streamed transfers stay off: diffusers only creates a CUDA stream, and the
    blocking path is valid on every backend.
    """
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(module, onload_device=device() if target_device is None else target_device,
                           offload_device=torch.device("cpu"), offload_type="leaf_level",
                           use_stream=False)


def offload_module(module, target_device=None):
    """Keep a whole component on CPU and move it in for each ``forward``.

    Returns the accelerate ``UserCpuOffloadHook``. Only use this for modules
    whose compute entry point is ``forward``; diffusers decoders expose
    ``encode``/``decode`` and need :func:`offload_leaves` instead.
    """
    from accelerate.big_modeling import cpu_offload_with_hook

    return cpu_offload_with_hook(module, execution_device=device() if target_device is None else target_device)


def configure_precision(matmul_allow_tf32: bool, cudnn_allow_tf32: bool) -> None:
    """Apply the fixed CUDA precision flags; NPU/CPU keep their backend defaults.

    On CUDA this is exactly ``torch.backends.cuda.matmul.allow_tf32`` +
    ``torch.backends.cudnn.allow_tf32``. Ascend precision is controlled by
    torch_npu/CANN, so no flags are touched there (and none exist on CPU builds).
    """
    if backend() != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(matmul_allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(cudnn_allow_tf32)
