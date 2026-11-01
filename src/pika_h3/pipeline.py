# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""One staged native Diffusers inference path for both released adapters."""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
KNOTS = [1000, 750, 500, 250, 100, 40, 0]
BUCKET_FILE = Path(__file__).parent / "core" / "buckets.yaml"


def _progress(stage, pct, **extra):
    """Machine-readable progress line for the WebUI (``flush=True``); logs stay human-readable."""
    fields = " ".join(f"{key}={value}" for key, value in extra.items())
    print(f"[progress] stage={stage} pct={int(pct)}" + (f" {fields}" if fields else ""), flush=True)


def resolve_canvas_args(args):
    """Resolve output ``width``/``height``/``num_frames`` from validated CLI args.

    ``--canvas WIDTHxHEIGHT`` bypasses the bucket table: ``cli.canvas_size`` has
    already checked both sides against the 32-px lattice and the aspect limit, so
    the explicit pair is used verbatim and no tier lookup happens. Otherwise the
    ``--resolution``/``--aspect`` row of ``buckets.yaml`` is selected exactly as
    before. Returns a plain dict so the geometry decision can be tested without
    the model stack.
    """
    from .core import geometry as G

    frames = G.align_num_frames(int(args.duration) * G.FPS)
    if getattr(args, "canvas", None) is not None:
        raw = args.canvas
        if isinstance(raw, str):
            # Accept a raw ``WIDTHxHEIGHT`` string from programmatic callers too.
            from .cli import canvas_size

            width, height = canvas_size(raw)
        else:
            width, height = int(raw[0]), int(raw[1])
        if width % G.CANVAS_MULTIPLE or height % G.CANVAS_MULTIPLE:
            raise ValueError(f"canvas must be a multiple of {G.CANVAS_MULTIPLE}, got {width}x{height}")
        if not 0.25 <= width / height <= 4.0:
            raise ValueError(f"canvas aspect ratio outside [0.25, 4.0], got {width}x{height}")
        return {"width": width, "height": height, "num_frames": frames, "tier": None,
                "resolution": "custom", "aspect": args.aspect, "canvas_source": "explicit"}
    import yaml

    table = yaml.safe_load(BUCKET_FILE.read_text())
    tier = "p" + args.resolution[:-1]
    width, height = table["tiers"][tier][table["aspects"].index(args.aspect)]
    return {"width": int(width), "height": int(height), "num_frames": frames, "tier": tier,
            "resolution": args.resolution, "aspect": args.aspect, "canvas_source": "bucket"}


def _tensor_parallel_rank(args):
    """Rank/world for weight tensor parallelism; ``(0, 1)`` when it is not requested."""
    world = int(getattr(args, "tensor_parallel", 1))
    if world <= 1:
        return 0, 1
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("hccl")
    if dist.get_world_size() != world:
        raise RuntimeError(f"--tensor-parallel {world} needs exactly {world} ranks; "
                           f"the process group reports {dist.get_world_size()}")
    return dist.get_rank(), world


THREADS_ENV = "PIKA_CPU_THREADS"
DEFAULT_CPU_THREADS = 16


def _limit_cpu_threads(max_threads: int | None = None) -> int:
    """Cap intra-op threads per rank, overridable with ``PIKA_CPU_THREADS``.

    ``torchrun`` only sets ``OMP_NUM_THREADS``; PyTorch's own intra-op pool still
    defaults to one thread per visible core, so two ranks each spawn hundreds of
    workers and starve each other (measured: sampling 43 s -> 76 s). The default
    cap is 16, further bounded by CPU affinity and the cgroup quota, because a
    machine swap can change both. Set ``PIKA_CPU_THREADS`` to retune.
    """
    import torch

    if max_threads is None:
        raw = os.environ.get(THREADS_ENV, "").strip()
        try:
            max_threads = int(raw) if raw else DEFAULT_CPU_THREADS
        except ValueError:
            raise ValueError(f"{THREADS_ENV} must be a positive integer, got {raw!r}") from None
        if max_threads < 1:
            raise ValueError(f"{THREADS_ENV} must be a positive integer, got {raw!r}")

    limit = max_threads
    try:
        limit = min(limit, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limit = min(limit, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError, IndexError):
        pass
    limit = max(1, int(limit))
    torch.set_num_threads(limit)
    return limit


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_digests(args):
    """SHA-256 of every generation input, captured once before the models load.

    The metadata block used to call ``file_sha256`` on the input paths after the
    whole run finished. A generation takes tens of minutes, so an input that was
    replaced or removed in the meantime made a *completed* run fail while writing
    its JSON: the validated MP4 was already on disk, but no ``output.json`` and no
    ``done`` marker were produced. Snapshot the digests up front instead, which
    also fails fast (before any model is loaded) when an input is missing.
    """
    return {
        "first_image": file_sha256(args.first_image) if args.first_image else None,
        "last_image": file_sha256(args.last_image) if args.last_image else None,
        "references": [file_sha256(spec.partition(":")[2]) for spec in args.ref],
    }


def resolve_assets(args):
    from huggingface_hub import snapshot_download
    task = "ref2va" if args.task == "ref2va" else "fl2va"
    if args.adapter_path:
        adapter_root = args.adapter_path
    else:
        root = snapshot_download(args.model_repo, revision=args.revision,
                                 allow_patterns=[f"{task}/*", "LICENSE", "NOTICE", "licenses/*"])
        adapter_root = Path(root) / task
    manifest = json.loads((adapter_root / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("rank") != 128 or manifest.get("alpha") != 128:
        raise ValueError("unsupported adapter format/rank/alpha")
    expected_policy = "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1"
    if manifest.get("numerics_policy") != expected_policy:
        raise ValueError("adapter numerical policy does not match this release")
    adapter = adapter_root / "adapter.safetensors"
    # Validate the exported release contract before loading any models.
    expected_trunk = "transformer_ref" if task == "ref2va" else "transformer"
    if manifest["trunk"] != expected_trunk:
        raise ValueError("adapter trunk does not match task")
    if manifest["sampling"]["knots"] != KNOTS or manifest["sampling"]["video_shift"] != 12 or manifest["sampling"]["audio_shift"] != 3 or manifest["sampling"]["method"] != "renoise":
        raise ValueError("unsupported adapter sampling contract")
    if file_sha256(adapter) != manifest["sha256"]:
        raise ValueError("adapter checksum mismatch")
    if manifest["base"] != {"repo_id": BASE_REPO, "revision": BASE_REVISION}:
        raise ValueError("adapter base identity mismatch")
    base = args.base_path or Path(snapshot_download(BASE_REPO, revision=BASE_REVISION,
        allow_patterns=[f"{expected_trunk}/*", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "audio_vae/*", "*.json", "LICENSE", "NOTICE", "licenses/*"]))
    return Path(base), adapter, manifest


def build_request(args, canvas, references, first_image=None, last_image=None):
    """Assemble the conditioning ``Request`` from resolved geometry and inputs.

    Pure (no model imports) so the ``--canvas`` plumbing can be tested without the
    model stack: ``pipeline.generate`` calls this with the output of
    :func:`resolve_canvas_args`.
    """
    from .core.conditioning import Request

    return Request(task=args.task, prompt=args.prompt,
        first_image=first_image, last_image=last_image,
        references=references, height=int(canvas["height"]), width=int(canvas["width"]),
        num_frames=int(canvas["num_frames"]),
        reference_rule="tiered", keyframe_fit="crop")


def generate(args):
    from .runtime import validate_runtime
    validate_runtime()
    import torch
    from PIL import Image
    from .core.conditioning import H3ConditionEncoder, Reference
    from .core.loader import build_transformer, apply_lora, load_lora_weights
    from .core.fused_ops import fuse_rmsnorm_modules
    from .core.numerics import prepare_conditioning, configure_numerics_backend
    from .core.schedules import H3SigmaSchedule
    from .core.evaluation import sample_evaluation_ladder
    from .core.wrapper import H3DiffusionWrapper
    from .core.vae import H3Decoders, write_mp4
    from .core import accelerator
    from .core import geometry as G

    if not accelerator.is_available():
        raise RuntimeError(f"No supported accelerator is available for model generation (backend={accelerator.backend()})")
    device = accelerator.device()
    tp_world = int(getattr(args, "tensor_parallel", 1))
    if tp_world > 1:
        # Pin this process to its own device before any HCCL collective. Without
        # this both ranks bind the default device and HCCL rejects the group with
        # "the same physical device ID" during communicator setup.
        device = torch.device(f"{device.type}:{int(os.environ['LOCAL_RANK'])}")
        getattr(torch, device.type).set_device(device)
    tp_rank, tp_world = _tensor_parallel_rank(args)
    # Applies to both paths: unconstrained PyTorch spawns one intra-op thread per
    # visible core (640 here), which hurts loading even in a single process.
    cpu_threads = _limit_cpu_threads()
    canvas = resolve_canvas_args(args)
    width, height, frames = canvas["width"], canvas["height"], canvas["num_frames"]
    _progress("prepare", 2, task=args.task, canvas=f"{width}x{height}", frames=frames)
    from .core.media import preflight_reference_specs
    _progress("loading", 5, what="references")
    # Snapshot the input digests before anything is decoded, and long before the
    # models run: a path that disappears or is replaced during a 30-minute
    # generation must not fail the JSON write of a finished run (see input_digests).
    digests = input_digests(args)
    preflight_reference_specs(args.ref)
    references = [Reference.from_spec(spec) for spec in args.ref]
    counts = {kind: sum(r.kind == kind for r in references) for kind in ("image", "video", "audio")}
    if len(references) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("reference limits: 9 images, 3 videos, 3 audio; 12 total")
    if args.task == "ref2va" and counts["image"] + counts["video"] == 0:
        raise ValueError("ref2va requires an image or video reference")
    request = build_request(
        args, canvas, references,
        first_image=Image.open(args.first_image).convert("RGB") if args.first_image else None,
        last_image=Image.open(args.last_image).convert("RGB") if args.last_image else None)
    start = time.monotonic()
    _progress("loading", 8, what="assets")
    base, adapter, manifest = resolve_assets(args)
    timings = {"assets_seconds": time.monotonic() - start}
    policy = manifest["numerics_policy"]
    configure_numerics_backend(policy, runtime="inference")
    # One flag drives every offload path: offloading only the transformer is not
    # enough when the text encoder alone (62 GiB) exceeds device memory.
    low_vram = bool(args.cpu_offload or args.low_vram)
    _progress("conditioning", 12, what="text-encoder")
    encoder = H3ConditionEncoder(str(base), device=device, text_encoder_checkpoint=str(args.text_encoder_path) if args.text_encoder_path else None, offload=low_vram)
    start = time.monotonic()
    accelerator.reset_peak_memory_stats()
    if tp_world > 1:
        # The text encoder and VAEs are replicated, so only the source rank runs
        # them; the others rebuild the deterministic skeleton and receive tensors.
        from .core.tensor_parallel import broadcast_conditioning

        if tp_rank == 0:
            cond = encoder.encode(request)
            cond = broadcast_conditioning(cond, src=0)
        else:
            cond = broadcast_conditioning(None, skeleton=encoder.skeleton(request), src=0)
    else:
        cond = encoder.encode(request)
    conditioning_peak = accelerator.max_memory_allocated()
    encoder.free()
    del encoder
    gc.collect()
    accelerator.empty_cache()
    timings["conditioning_seconds"] = time.monotonic() - start
    _progress("conditioning", 40, what="done")
    start = time.monotonic()
    _progress("loading", 45, what="transformer")
    model = build_transformer(str(base), dtype=torch.bfloat16, device="cpu", subfolder=manifest["trunk"])
    apply_lora(model, rank=manifest["rank"], alpha=manifest["alpha"], target_modules=manifest.get("target_modules"), master_dtype=torch.bfloat16)
    load_lora_weights(model, str(adapter))
    model.requires_grad_(False).eval()
    # Opt-in only: the fused kernel is numerically equivalent to ATen within bf16
    # rounding (0.2 % per block, i.e. below one ulp), but a 1-ulp difference per
    # block is amplified by six-step consistency sampling into a visibly different
    # video at the same seed. The released contract is bit-exact output, so the
    # fusion is off unless explicitly requested.
    fused_norms = fuse_rmsnorm_modules(model) if os.environ.get("PIKA_FUSED_RMSNORM", "0") == "1" else 0
    tp_report = None
    if tp_world > 1:
        # Shard the merged base weights AND the adapter factors together: the shard
        # modules read `.weight` directly, so an adapter left inside a PEFT wrapper
        # would be dropped rather than split (see tensor_parallel.install_tensor_parallel).
        from .core.tensor_parallel import install_tensor_parallel
        tp_report = install_tensor_parallel(model, tp_rank, tp_world)
    if low_vram:
        # Adaptive placement: keep the weights resident when the shard **and this
        # request's activations** fit (a TP shard usually does), otherwise stream
        # blocks from pinned host memory. Both branches compute the same function,
        # so the output does not depend on the choice -- measured bit-identical mp4
        # for TP2 at 480p/5 s.
        transformer_offloaded = accelerator.enable_group_offload(model, onload_device=device,
            offload_device=torch.device("cpu"), num_blocks_per_group=1,
            use_stream=True, record_stream=False,
            sequence_length=int(cond["packed_tokens"]), hidden_size=int(model.config.hidden_size))
        wrapper = H3DiffusionWrapper(model).eval()
    else:
        transformer_offloaded = False
        wrapper = H3DiffusionWrapper(model).eval().to(device)
    timings["transformer_load_seconds"] = time.monotonic() - start
    _progress("loading", 55, what="transformer-done")
    geom = cond["geometry"]
    schedule = H3SigmaSchedule({"denoising_step_list": KNOTS, "video_score_sigma_shift":12, "audio_score_sigma_shift":3}, device=device)
    start = time.monotonic()
    accelerator.reset_peak_memory_stats()
    _progress("sampling", 60, step=0, total=len(KNOTS) - 1)

    def _sampling_step(index):
        # Called by the sampler before each generator evaluation; the marker keeps
        # the coarse 60 -> 82 band but reports the real step index.
        total = len(KNOTS) - 1
        done = min(total, int(index) + 1)
        _progress("sampling", 60 + round(22 * done / total), step=done, total=total)
        return {}

    with torch.inference_mode():
        video, audio = sample_evaluation_ladder(wrapper,
            conditional_dict=prepare_conditioning(cond, device, policy=policy),
            video_shape=[1,geom.num_latent_frames,G.VIDEO_LATENT_CHANNELS,geom.latent_height,geom.latent_width],
            audio_shape=[1,geom.num_audio_latents,G.AUDIO_EXTERNAL_DIM],
            sigmas_v=schedule.video_ladder, sigmas_a=schedule.audio_ladder,
            seed=args.seed, device=device, step_kwargs=_sampling_step)
    video, audio = video[0].cpu(), audio[0].cpu()
    accelerator.synchronize()
    timings["sampling_seconds"] = time.monotonic() - start
    _progress("sampling", 82, step=len(KNOTS) - 1, total=len(KNOTS) - 1)
    peak = accelerator.max_memory_allocated()
    del wrapper, model, cond
    gc.collect()
    accelerator.empty_cache()
    if tp_world > 1:
        # Both ranks sampled the same latent; only rank 0 decodes and writes, and
        # the second rank waits so the collective order stays fixed.
        import torch.distributed as dist
        if tp_rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return
    start = time.monotonic()
    accelerator.reset_peak_memory_stats()
    _progress("decode", 85, what="decoders")
    decoders = H3Decoders(str(base), device=device, offload=low_vram)
    with torch.inference_mode():
        _progress("decode", 88, what="video")
        pixels = decoders.decode_video(video)
        _progress("decode", 93, what="audio")
        wave = decoders.decode_audio(audio)
    decode_peak = accelerator.max_memory_allocated()
    if pixels.shape[:3] != (frames, height, width):
        raise RuntimeError(f"decoded geometry mismatch: {pixels.shape}")
    import numpy as np
    if not np.isfinite(wave).all():
        raise RuntimeError("decoded audio is not finite")
    wave = wave[:, :round(frames / G.FPS * G.AUDIO_SAMPLE_RATE)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _progress("encode", 95, what="mp4")
    write_mp4(str(args.output), pixels, audio=wave)
    import av
    with av.open(str(args.output)) as encoded:
        stream = encoded.streams.video[0]
        if (stream.width,stream.height) != (width,height) or not encoded.streams.audio:
            raise RuntimeError("encoded video/audio validation failed")
        audio_stream = encoded.streams.audio[0]
        if stream.average_rate != G.FPS or audio_stream.codec_context.sample_rate != G.AUDIO_SAMPLE_RATE or audio_stream.codec_context.channels != 2:
            raise RuntimeError("encoded frame rate or stereo audio format mismatch")
        if sum(1 for _ in encoded.decode(video=0)) != frames:
            raise RuntimeError("encoded frame count mismatch")
    timings["decode_encode_seconds"] = time.monotonic() - start
    metadata = {"task": args.task, "prompt": args.prompt, "seed": args.seed,
        "resolution": canvas["resolution"], "aspect": canvas["aspect"], "requested_seconds": args.duration,
        "canvas": {"width": width, "height": height, "num_frames": frames,
                   "source": canvas["canvas_source"], "bucket_tier": canvas["tier"]},
        "geometry": asdict(geom), "adapter_sha256": manifest["sha256"],
        "adapter_revision": args.revision, "base": manifest["base"],
        "local_base_override": args.base_path is not None,
        "transformer_cpu_offload": args.cpu_offload,
        "transformer_offloaded": transformer_offloaded,
        "low_vram": low_vram,
        "component_cpu_offload": low_vram,
        "local_conditioner_override": args.text_encoder_path is not None,
        "bucket_sha256": file_sha256(BUCKET_FILE), "noise_version": "pika-flex-noise-v1",
        "numerics_policy":policy, "sampling":manifest["sampling"],
        "fused_rmsnorm_modules": fused_norms,
        "timings":timings, "sampling_peak_allocated_bytes":peak,
        "conditioning_peak_allocated_bytes": conditioning_peak,
        "decode_peak_allocated_bytes": decode_peak,
        "tensor_parallel_world_size": tp_world,
        "tensor_parallel_rank": tp_rank,
        "tensor_parallel_shard_bytes": None if tp_report is None else tp_report["local_shard_bytes"],
        "tensor_parallel_fraction": None if tp_report is None else round(tp_report["per_rank_fraction"], 4),
        "torch":torch.__version__, "gpu":accelerator.device_name(),
        "reference_order": [r.kind for r in references]}
    import importlib.metadata
    metadata["dependencies"] = {name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "peft", "av", "safetensors")}
    # Digest computed before the models loaded: the input files may already be gone
    # (or replaced) by now, which must not fail a finished generation.
    metadata["input_sha256"] = digests
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    _progress("done", 100, output=str(args.output))
    print(f"Saved {args.output}", flush=True)
    if tp_world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
