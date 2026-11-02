# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Inference-only weight tensor parallelism for the H3 transformer (HCCL, 2 ranks).

Each block's attention heads and feed-forward intermediate channels are split
across ranks; the residual stream stays replicated, so no sequence-level
collective is needed and the model needs no attention-kernel changes.

Collectives per block, per forward:
  * column-sharded QKV: no collective (each rank owns whole heads);
  * row-sharded attention output: one FP32 ``all_reduce``;
  * column-sharded SwiGLU input: no collective;
  * row-sharded FFN output: one FP32 ``all_reduce``;
  * column-sharded AdaLN projection: ``all_gather`` of the small modulation table,
    because every rank needs the complete table to modulate a full residual row.

Design notes mirroring the vdn-minimax-h3 ``ascend-npu-a3`` branch:

* Sharding happens on CPU; every retained parameter is a clone, so no rank
  keeps the full source storage. LoRA factors are sliced alongside the base
  weights (:class:`_LoraFactors`), because the shard modules read ``.weight``
  directly and never run a PEFT wrapper's ``forward``.
* ``RowLinear`` accumulates in FP32 before the cast back, which is what keeps the
  split-K reduction close to the single-card result. With BF16 accumulation the
  drift is an order of magnitude larger.
* Only the block stack is sharded. The patch projections, token refiner, AdaLN
  time embedding and output heads are small and stay replicated on every rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .accelerator import device as _device
from . import accelerator


def shard_interval(size: int, rank: int, world_size: int) -> slice:
    """Contiguous shard; ``size`` must divide evenly by ``world_size``.

    Equally sized shards keep every rank's kernel shapes identical, which is what
    the HCCL collectives and the per-rank memory budget assume.
    """
    if size % world_size:
        raise ValueError(f"dimension {size} is not divisible by tp world size {world_size}")
    step = size // world_size
    return slice(rank * step, (rank + 1) * step)


class _LoraFactors(nn.Module):
    """Low-rank adapter factors carried next to a sharded base weight.

    PEFT applies ``base(x) + scaling * B(A(x))`` inside the wrapper's ``forward``.
    The shard modules replace that wrapper and read ``.weight`` directly, so the
    factors have to travel with the shard and be applied by the shard's own
    forward. Slicing the factors exactly like the base weight keeps the result
    equal to the unsharded path: every output row of ``B(A(x))`` is independent,
    and ``RowLinear`` reduces the input-channel split in FP32.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, scaling: float):
        super().__init__()
        self.a = nn.Parameter(a.clone(), requires_grad=False)  # [r, in]
        self.b = nn.Parameter(b.clone(), requires_grad=False)  # [out, r]
        self.scaling = float(scaling)

    def forward(self, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        adapter_dtype = self.a.dtype
        hidden = F.linear(x.to(adapter_dtype), self.a)
        return (F.linear(hidden, self.b) * self.scaling).to(out_dtype)


def adapter_factors(linear):
    """``(A, B, scaling)`` of a single-adapter PEFT LoRA layer, else ``None``.

    Only vanilla LoRA is accepted: a variant (DoRA, LoRA-FA, ...) changes the
    arithmetic in ways the shard modules do not reproduce, so it is refused
    rather than silently approximated.
    """
    from peft.tuners.lora.layer import LoraLayer

    if not isinstance(linear, LoraLayer):
        return None
    active = list(linear.active_adapters)
    if len(active) != 1:
        raise ValueError(f"tensor parallel needs exactly one active adapter, got {active}")
    name = active[0]
    if name in linear.lora_variant:
        raise ValueError(f"tensor parallel does not support the {type(linear.lora_variant[name]).__name__} variant")
    if linear.lora_bias.get(name, False):
        raise ValueError("tensor parallel does not support LoRA with a bias")
    return (linear.lora_A[name].weight, linear.lora_B[name].weight, linear.scaling[name])


MAX_WORLD_SIZE = 8


class ColumnLinear(nn.Module):
    """Output-feature shard. Produces this rank's slice of the full output."""

    def __init__(self, linear: nn.Linear, rows: slice | torch.Tensor, gather: bool = False, adapter=None):
        super().__init__()
        self.weight = nn.Parameter(linear.weight[rows].clone(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias[rows].clone(), requires_grad=False) if linear.bias is not None else None
        self.in_features = linear.in_features
        self.out_features = self.weight.shape[0]
        self.gather = gather
        self.lora = None
        if adapter is not None:
            a, b, scaling = adapter
            self.lora = _LoraFactors(a, b[rows], scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight, self.bias)
        if self.lora is not None:
            local = local + self.lora(x, local.dtype)
        if not self.gather:
            return local
        # Small tables (AdaLN) must be identical and complete on every rank.
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts, dim=-1)


class RowLinear(nn.Module):
    """Input-feature shard, reduced across ranks in FP32 before the cast back."""

    def __init__(self, linear: nn.Linear, columns: slice, adapter=None):
        super().__init__()
        self.weight = nn.Parameter(linear.weight[:, columns].clone(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.clone(), requires_grad=False) if linear.bias is not None else None
        self.in_features = self.weight.shape[1]
        self.out_features = linear.out_features
        self.lora = None
        if adapter is not None:
            a, b, scaling = adapter
            self.lora = _LoraFactors(a[:, columns], b, scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora is None:
            total = F.linear(x, self.weight).float()
        else:
            total = F.linear(x, self.weight).float() + self.lora(x, torch.float32)
        dist.all_reduce(total)
        total = total.to(self.weight.dtype)
        return total if self.bias is None else total + self.bias


def merge_adapters(model) -> int:
    """Fold every PEFT LoRA delta into its base weight; return the module count.

    Kept as the fallback for layers whose shard cannot carry the factors. The
    sharding path itself preserves adapters exactly (see ``_LoraFactors``), so
    this is only used by callers that want an adapter-free model.
    """
    from peft.tuners.lora.layer import LoraLayer

    merged = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        for name in list(module.active_adapters):
            try:
                module.merge(adapter_names=[name])
            except NotImplementedError as exc:
                raise RuntimeError(
                    f"tensor parallel needs every LoRA layer mergeable, {type(module).__name__} is not") from exc
        merged += 1
    return merged


def install_tensor_parallel(model, rank: int, world_size: int) -> dict:
    """Replace the block projections in place; call on CPU before device placement.

    Adapter factors are sliced alongside the base weights, so the sharded model
    computes the same function as the unsharded one. Returns a small report
    describing the split so the caller can log it.
    """
    if getattr(model, "_pika_tensor_parallel", None) is not None:
        raise RuntimeError("tensor parallel is already installed")
    if world_size < 2:
        raise ValueError("tensor parallel needs at least two ranks")
    # The split is uniform over heads and FFN channels, so any world size that
    # divides both works; two ranks is what has been measured on this machine.
    if not 1 <= world_size <= MAX_WORLD_SIZE:
        raise ValueError(f"tensor parallel supports 1-{MAX_WORLD_SIZE} ranks, got {world_size}")
    if model.training:
        raise ValueError("tensor parallel is inference-only; call model.eval() first")

    before = sum(p.numel() * p.element_size() for p in model.parameters())
    with_adapters = 0
    sharded = set()
    layout = []

    for block in model.transformer_blocks:
        attn = block.attn
        heads = shard_interval(attn.heads, rank, world_size)
        channels = slice(heads.start * attn.head_dim, heads.stop * attn.head_dim)
        local_heads = heads.stop - heads.start
        layout.append({"global_heads": attn.heads, "local_heads": local_heads,
                       "head_dim": attn.head_dim,
                       "global_ffn": block.ff.net[2].in_features,
                       "local_ffn": block.ff.net[2].in_features // world_size})

        # Attention: Q/K/V own whole heads, so their outputs are already sharded.
        for name in ("to_q", "to_k", "to_v"):
            linear = getattr(attn, name)
            adapter = adapter_factors(linear)
            with_adapters += adapter is not None
            setattr(attn, name, ColumnLinear(linear, channels, adapter=adapter))
        attn.heads = local_heads
        attn.inner_dim = local_heads * attn.head_dim
        # The attention output projects back to the full width: row-shard + all_reduce.
        adapter = adapter_factors(attn.to_out[0])
        with_adapters += adapter is not None
        attn.to_out[0] = RowLinear(attn.to_out[0], channels, adapter=adapter)

        for module in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]):
            sharded.add(id(module.weight))
            if getattr(module, "bias", None) is not None:
                sharded.add(id(module.bias))

        # Feed-forward: SwiGLU stores [value, gate] concatenated, so shard matching
        # slices of both halves and keep the interleaving identical to the source.
        ff = block.ff
        intermediate = ff.net[2].in_features
        chunk = intermediate // world_size
        rows = torch.cat((torch.arange(rank * chunk, (rank + 1) * chunk),
                          torch.arange(intermediate + rank * chunk, intermediate + (rank + 1) * chunk)))
        adapter = adapter_factors(ff.net[0].proj)
        with_adapters += adapter is not None
        ff.net[0].proj = ColumnLinear(ff.net[0].proj, rows, adapter=adapter)
        adapter = adapter_factors(ff.net[2])
        with_adapters += adapter is not None
        ff.net[2] = RowLinear(ff.net[2], shard_interval(intermediate, rank, world_size), adapter=adapter)

        for module in (ff.net[0].proj, ff.net[2]):
            sharded.add(id(module.weight))
            if getattr(module, "bias", None) is not None:
                sharded.add(id(module.bias))

        # AdaLN projects the shared time embedding to all six modulation tables;
        # every rank needs the complete table for its full-width residual rows.
        adapter = adapter_factors(block.adaln_proj.linear)
        with_adapters += adapter is not None
        block.adaln_proj.linear = ColumnLinear(
            block.adaln_proj.linear,
            shard_interval(block.adaln_proj.linear.out_features, rank, world_size),
            gather=True, adapter=adapter)
        sharded.add(id(block.adaln_proj.linear.weight))
        if block.adaln_proj.linear.bias is not None:
            sharded.add(id(block.adaln_proj.linear.bias))

    after = sum(p.numel() * p.element_size() for p in model.parameters())
    report = {"rank": rank, "world_size": world_size,
              "merged_adapters": with_adapters,
              "parameter_bytes_before": before, "parameter_bytes_after": after,
              "per_rank_fraction": after / before if before else 0.0,
              "local_shard_bytes": after,
              "block_layout": layout}
    model._pika_tensor_parallel = report
    return report


def _flatten_conditioning(cond, prefix="", out=None):
    """Collect ``(path, tensor)`` for every tensor in the nested conditioning dict.

    Only dict/list containers are traversed; dataclass leaves (geometry) are kept
    as opaque objects and shipped through the object broadcast.
    """
    out = [] if out is None else out
    if torch.is_tensor(cond):
        out.append((prefix, cond))
    elif isinstance(cond, dict):
        for key, value in cond.items():
            _flatten_conditioning(value, f"{prefix}\x00{key}", out)
    elif isinstance(cond, list):
        for index, value in enumerate(cond):
            _flatten_conditioning(value, f"{prefix}\x00{index}", out)
    return out


def _rebuild_conditioning(cond, tensors):
    """Unused: kept only for reference; see :func:`broadcast_conditioning`."""
    return cond


def _scalar_leaves(node, prefix="", out=None):
    """Collect ``(path, value)`` for every non-tensor, non-container leaf."""
    out = [] if out is None else out
    if torch.is_tensor(node):
        return out
    if isinstance(node, dict):
        for key, value in node.items():
            _scalar_leaves(value, f"{prefix}\x00{key}", out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _scalar_leaves(value, f"{prefix}\x00{index}", out)
    else:
        out.append((prefix, node))
    return out


def _assign_path(root, path, value):
    """Place a scalar leaf back at ``path``.

    Paths are built by :func:`_scalar_leaves` and :func:`_flatten_conditioning`
    as ``\\x00``-separated steps; a step indexes a list when the current node is
    a list and keys a dict otherwise.
    """
    steps = [step for step in path.split("\x00") if step]
    node = root
    for step in steps[:-1]:
        node = node[int(step)] if isinstance(node, list) else node[step]
    if not steps:
        return value
    if isinstance(node, list):
        node[int(steps[-1])] = value
    else:
        node[steps[-1]] = value
    return root


def broadcast_conditioning(cond, skeleton=None, src: int = 0) -> dict:
    """Make the text/VAE conditioning byte-identical on every rank.

    The encoder runs once on ``src``; every other rank receives exactly those
    elements. This is deterministic, unlike re-encoding on each rank.

    ``skeleton`` is the same container with non-tensor leaves already built
    deterministically on every rank; only tensors and their paths travel over
    the wire, plus scalar leaves such as ``packed_tokens``.
    """
    world = dist.get_world_size()
    if world <= 1:
        return cond
    rank = dist.get_rank()

    flat = _flatten_conditioning(cond) if rank == src else None
    header = None
    if rank == src:
        header = {"paths": [p for p, _ in flat],
                  "specs": [{"shape": [int(s) for s in t.shape], "dtype": str(t.dtype)} for _, t in flat],
                  "scalars": _scalar_leaves(cond)}
    box = [header]
    dist.broadcast_object_list(box, src=src)
    header = box[0]

    if rank == src:
        tensors = [t.contiguous() for _, t in flat]
    else:
        tensors = []
        for spec in header["specs"]:
            dtype = getattr(torch, spec["dtype"].split(".", 1)[1])
            tensors.append(torch.empty(tuple(spec["shape"]), dtype=dtype, device="cpu"))

    # HCCL only broadcasts device tensors, so stage each payload on this rank's
    # device and bring it back to CPU afterwards.
    local_device = _device()
    for tensor in tensors:
        staged = tensor.to(local_device, non_blocking=False)
        dist.broadcast(staged, src=src)
        if rank != src:
            tensor.copy_(staged.cpu())
        del staged
    accelerator.empty_cache()

    if rank == src:
        return cond
    if skeleton is None:
        raise ValueError("non-source ranks must pass the deterministic skeleton")
    # Paths came from the source, so assign by path rather than by traversal
    # order: a skeleton cannot distinguish a tensor placeholder from a real
    # ``None`` scalar.
    rebuilt = skeleton
    for (path, _), tensor in zip(zip(header["paths"], header["specs"]), tensors):
        rebuilt = _assign_path(rebuilt, path, tensor)
    for path, value in header["scalars"]:
        rebuilt = _assign_path(rebuilt, path, value)
    return rebuilt

