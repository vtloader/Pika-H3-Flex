# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
# Portions adapted from Diffusers MiniMax H3: decoders.py.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# The adapted upstream portions are licensed under Apache-2.0; see licenses/Apache-2.0.txt.
# Modified by Mellis, Inc. in 2026. Copyright © 2026 Mellis, Inc.
# Model Derivative of MiniMax H3, distributed under the MiniMax H3 Community License Agreement.
# Code remains licensed under Apache-2.0; the model terms above apply to the adapters.
# Changes: Adapted normalization and layout handling for standalone VAE decoding.
# Source revision: 3a2f35d4efa4c059c8bfb3bc0d6c906264895c81.

"""H3 VAE decode helpers for validation clips (latents are stored/normalized the pipeline's way)."""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import torch

from . import accelerator
from . import geometry as G

# Decode precision. ``fp32`` reproduces the fp32-weight arithmetic exactly;
# ``fp16`` is the recipe the released model was validated with upstream (the
# official pipeline autocasts the same fp32 weights to fp16 on CUDA) and uses
# the NPU's fast cube path.
DECODE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DECODE_DTYPE_ENV = "PIKA_VAE_DECODE_DTYPE"

# Spatial tiles evaluated per device forward. Tiles are independent ViT
# forwards of identical shape, so batching them turns many small launches into
# fewer large ones. ``1`` keeps the exact per-tile arithmetic; batching only
# reorders the attention reduction, which moves a few pixels by one 8-bit level.
TILE_BATCH_ENV = "PIKA_VAE_TILE_BATCH"


def decode_dtype(default: torch.dtype = torch.float32) -> torch.dtype:
    """Configured decode precision; ``PIKA_VAE_DECODE_DTYPE`` overrides ``default``."""
    raw = os.environ.get(DECODE_DTYPE_ENV, "").strip().lower()
    if not raw:
        return default
    if raw not in DECODE_DTYPES:
        raise ValueError(f"{DECODE_DTYPE_ENV} must be one of {sorted(DECODE_DTYPES)}, got {raw!r}")
    return DECODE_DTYPES[raw]


def tile_batch_size(default: int = 1) -> int:
    """How many spatial tiles the decoder evaluates per forward."""
    raw = os.environ.get(TILE_BATCH_ENV, "").strip()
    if not raw:
        return max(1, int(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{TILE_BATCH_ENV} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{TILE_BATCH_ENV} must be a positive integer, got {raw!r}")
    return value


def chunk_ranges(num_chunks: int, world: int) -> list[tuple[int, int]]:
    """Split ``[0, num_chunks)`` into ``world`` contiguous half-open ranges.

    Ranks get either ``num_chunks // world`` or one more chunk, with the extra
    chunks going to the lowest ranks, so rank 0 (which also encodes) is never the
    last one to finish. Empty ranges are returned for extra ranks.
    """
    world = max(1, int(world))
    base, extra = divmod(max(0, int(num_chunks)), world)
    ranges, start = [], 0
    for rank in range(world):
        size = base + (1 if rank < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


class H3Decoders:
    def __init__(self, checkpoint_dir: str, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                 offload: bool = False, compute_dtype: torch.dtype | None = None):
        # The released VAEs keep fp32 biases/norms; the pipeline runs them in fp32 (10 GB video VAE) — keep that.
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=dtype)
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(checkpoint_dir, subfolder="audio_vae", torch_dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        # ``fp32`` weights stay fp32; the compute dtype only selects autocast.
        self.compute_dtype = decode_dtype(torch.float32) if compute_dtype is None else compute_dtype
        # Batching tiles reorders the attention reduction by one rounding step
        # (<= 1/255 on delivered frames), so it is only the default when the
        # decode is already approximate; fp32 stays per-tile and bit-exact.
        # Measured at 768p: fp16 19.4 s (batch 1) -> 15.4 s (batch 4).
        self.tile_batch = tile_batch_size(4 if self.compute_dtype != torch.float32 else 1)
        if offload:
            # Video VAE is 9.7 GiB fp32: keep it on CPU and move it in whole around
            # the decode stage. Leaf hooks are unsafe for it, and accelerate's module
            # hook only fires on ``forward`` while diffusers calls ``decode``.
            # The audio VAE is 0.56 GiB and uses ``weight_norm`` weights, so it stays
            # resident on the device instead of being shuttled.
            self._offload = True
            self.audio_vae = self.audio_vae.to(self.device)
        else:
            self._offload = False
            self.vae = self.vae.to(self.device)
            self.audio_vae = self.audio_vae.to(self.device)
        self.vae = self.vae.eval()
        self.audio_vae = self.audio_vae.eval()
        self.video_mean = torch.tensor(self.vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
        self.video_std = torch.tensor(self.vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
        self.audio_mean = torch.tensor(self.audio_vae.config.latents_mean, device=self.device).view(1, -1, 1)
        self.audio_std = torch.tensor(self.audio_vae.config.latents_std, device=self.device).view(1, -1, 1)
        self._pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(3, 1, 1, 1)
        self._pixel_std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(3, 1, 1, 1)

    def to(self, device) -> "H3Decoders":
        self.device = torch.device(device)
        self.vae.to(device)
        self.audio_vae.to(device)
        self.video_mean = self.video_mean.to(device)
        self.video_std = self.video_std.to(device)
        self.audio_mean = self.audio_mean.to(device)
        self.audio_std = self.audio_std.to(device)
        self._pixel_mean = self._pixel_mean.to(device)
        self._pixel_std = self._pixel_std.to(device)
        return self

    # ---- video -----------------------------------------------------------------------------------------------
    def chunk_layout(self, num_latent_frames: int) -> dict:
        """Temporal chunk geometry of ``vae._decode`` for this latent length.

        Exposed so a caller can split the clip across ranks (or resume a decode)
        without guessing how upstream pads and overlaps its chunks.
        """
        vae = self.vae
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        temporal_ratio = vae.temporal_compression_ratio
        num_tokens = int(num_latent_frames) + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        frames_per_chunk = tokens_chunk_size * temporal_ratio - vae.frame_pre_padding
        # Chunks 0..n-1 each emit ``frames_per_chunk``; the trailing overlap block
        # only covers ``token_overlap`` latent frames, so it is shorter by exactly
        # ``frame_overlap``. (37 latents -> 7*17 + 5 = 124, 72 -> 14*17 + 5 = 243.)
        return {"chunks": num_chunks, "pad_tokens": pad_tokens,
                "frames_per_chunk": frames_per_chunk, "frame_overlap": vae.frame_overlap,
                "frames": num_chunks * frames_per_chunk + vae.frame_overlap}

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        """One temporal clip, spatial tiles evaluated in device batches.

        Upstream loops over the tiles one by one. Each tile is an independent
        ViT forward of the same shape, so stacking ``tile_batch`` of them turns
        T small launches into one larger one. ``PIKA_VAE_TILE_BATCH=1`` restores
        the per-tile arithmetic, which is bit-identical to upstream.
        """
        vae = self.vae
        if not vae.use_tiling:
            return vae.decoder(vae.post_quant_conv(z))
        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_idx, y_len, y_ov = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
        ratio = vae.spatial_compression_ratio
        tiles = [z[..., i // ratio:i // ratio + i_len // ratio, j // ratio:j // ratio + j_len // ratio]
                 for i, i_len in zip(y_idx, y_len) for j, j_len in zip(x_idx, x_len)]
        outputs = []
        for start in range(0, len(tiles), self.tile_batch):
            batch = torch.cat(tiles[start:start + self.tile_batch], dim=0)
            outputs.append(vae.decoder(vae.post_quant_conv(batch)))
        decoded = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        rows, index = [], 0
        for _ in y_idx:
            row = []
            for _ in x_idx:
                row.append(decoded[index:index + 1])
                index += 1
            rows.append(row)
        return vae._stitch_tiles(rows, y_ov, x_ov)

    def _decode_chunks(self, z: torch.Tensor, num_latent_frames: int,
                       first: int = 0, last: int | None = None) -> Iterator[torch.Tensor]:
        """Yield pixel chunks of output chunks ``[first, last)``, in order.

        Mirrors ``vae._decode`` exactly (same padding, same per-chunk slicing,
        same linear cross-fade, same tail trim) but hands each chunk to the
        caller instead of keeping the whole clip: at 1080p/10 s the concatenated
        fp32 clip is ~22 GiB for a 1.5 GiB result. Chunks tile the clip without
        gaps, so a caller can encode them as they arrive.

        ``first``/``last`` select a contiguous output range for sharded decoding.
        One chunk before ``first`` is decoded as well, because the cross-fade of
        the first emitted chunk needs its tail; that is the only duplicated work.
        """
        vae = self.vae
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        temporal_ratio = vae.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        layout = self.chunk_layout(num_latent_frames)
        num_chunks = layout["chunks"]
        last = num_chunks if last is None else min(int(last), num_chunks)
        first = max(0, int(first))
        if first >= last:
            return
        if layout["pad_tokens"]:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, layout["pad_tokens"], 1, 1)], dim=2)

        # Pixel frames produced by the repeated padding latents: upstream drops
        # them from the very end of the clip, i.e. from the final emitted chunks.
        # ``num_tokens_before_pad`` is measured on the *padded* tensor upstream
        # (``z.shape[2] - pad_tokens``), which is just the original latent count.
        intra_tail = vae.config.clip_length % temporal_ratio
        num_tokens_before_pad = int(num_latent_frames)
        pad_frames = sum(intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                         else temporal_ratio for k in range(layout["pad_tokens"]))

        pending = None  # held back so the very last chunk can be trimmed

        overlap = None
        # Chunk ``first - 1`` is decoded for its tail but never emitted.
        for i in range(max(0, first - 1), last):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start:start + tokens_chunk_size + vae.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    if i >= first:
                        if pending is not None:
                            yield pending
                        pending = chunk
                    else:
                        del chunk
                else:
                    overlap = chunk
            del clip
        if overlap is not None and last >= num_chunks:
            # The trailing half-chunk completes the clip when the range reaches the end.
            tail = [pending, overlap] if pending is not None else [overlap]
            overlap = None
        else:
            tail = [pending] if pending is not None else []
        if pad_frames and last >= num_chunks:
            # Trim from the very end of the clip, which may cross a chunk boundary.
            remaining = pad_frames
            while tail and remaining:
                last_chunk = tail[-1]
                drop = min(remaining, last_chunk.shape[2])
                tail[-1] = last_chunk[:, :, :last_chunk.shape[2] - drop]
                remaining -= drop
                if not tail[-1].shape[2]:
                    tail.pop()
        for chunk in tail:
            if chunk.shape[2]:
                yield chunk

    def _pixels(self, video: torch.Tensor) -> np.ndarray:
        """``[1, 3, N, H, W]`` VAE space -> uint8 ``[N, H, W, 3]``."""
        # Match the official pipeline unconditionally; bright decoded tensors
        # may contain no value below -0.5 and still use ImageNet normalization.
        video = video[0].float() * self._pixel_std + self._pixel_mean
        return video.clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()

    def decode_video_chunks(self, latents_external: torch.Tensor, *, first: int = 0,
                            last: int | None = None) -> Iterator[np.ndarray]:
        """Yield ``uint8`` frame blocks covering output chunks ``[first, last)``.

        ``latents_external`` is ``[F, 24, H, W]`` normalized. Blocks arrive in
        order and tile the clip without gaps, so a caller can encode them as they
        are produced instead of holding the full clip. ``first``/``last`` let a
        sharded caller decode one contiguous range per rank (see
        :func:`chunk_ranges` and :meth:`chunk_layout`).

        ``torch.no_grad`` is entered inside the generator body on purpose: a
        decorator on a generator only covers the call that *creates* it, so the
        whole decode would otherwise build an autograd graph over 36 blocks.
        """
        with torch.no_grad():
            yield from self._decode_video_chunks(latents_external, first, last)

    def _decode_video_chunks(self, latents_external: torch.Tensor, first: int,
                             last: int | None) -> Iterator[np.ndarray]:
        lat = latents_external.to(self.device, torch.float32).permute(1, 0, 2, 3).unsqueeze(0)
        lat = lat * self.video_std + self.video_mean
        num_latent_frames = lat.shape[2]
        if self._offload:
            self.vae.to(self.device)
        autocast = torch.autocast(device_type=self.device.type, dtype=self.compute_dtype,
                                  enabled=self.compute_dtype != torch.float32)
        try:
            with autocast:
                for chunk in self._decode_chunks(lat.to(self.dtype), num_latent_frames, first, last):
                    pixels = self._pixels(chunk)
                    del chunk
                    yield pixels
        finally:
            if self._offload:
                self.vae.to("cpu")
                accelerator.empty_cache()

    @torch.no_grad()
    def decode_video(self, latents_external: torch.Tensor) -> np.ndarray:
        """``[F, 24, H, W]`` normalized -> uint8 frames ``[N, H*16, W*16, 3]``."""
        parts = list(self.decode_video_chunks(latents_external))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    @torch.no_grad()
    def decode_audio(self, latents_external: torch.Tensor) -> np.ndarray:
        """``[T, 64]`` normalized -> float32 waveform ``[2, S]`` at 32 kHz."""
        from .packing import audio_external_to_storage

        lat = audio_external_to_storage(latents_external.to(self.device, torch.float32))  # [2, 32, T]
        lat = lat * self.audio_std + self.audio_mean
        audio = self.audio_vae.decode(lat.to(self.dtype), return_dict=False)[0]
        audio = audio.float().reshape(G.AUDIO_CHANNELS, -1).clamp(-1, 1).cpu().numpy()
        return audio


class _Mp4Writer:
    """Incremental H.264/AAC writer: frames are muxed as they arrive."""

    def __init__(self, path: str, width: int, height: int, fps: int = G.FPS,
                 audio: Optional[np.ndarray] = None, sample_rate: int = G.AUDIO_SAMPLE_RATE):
        import av

        self.container = av.open(path, mode="w")
        self.vstream = self.container.add_stream("libx264", rate=fps)
        self.vstream.width = int(width)
        self.vstream.height = int(height)
        self.vstream.pix_fmt = "yuv420p"
        self.audio = audio
        self.sample_rate = sample_rate
        self.astream = None
        if audio is not None:
            self.astream = self.container.add_stream("aac", rate=sample_rate)
            self.astream.layout = "stereo"
        self.frames = 0

    def write(self, frames: np.ndarray) -> None:
        import av

        for frame in frames:
            packet = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for encoded in self.vstream.encode(packet):
                self.container.mux(encoded)
            self.frames += 1

    def close(self) -> None:
        import av

        for packet in self.vstream.encode():
            self.container.mux(packet)
        if self.astream is not None:
            pcm = np.ascontiguousarray(self.audio.astype(np.float32))
            chunk = 1024
            for start in range(0, pcm.shape[1], chunk):
                af = av.AudioFrame.from_ndarray(pcm[:, start:start + chunk], format="fltp", layout="stereo")
                af.sample_rate = self.sample_rate
                af.pts = start
                for packet in self.astream.encode(af):
                    self.container.mux(packet)
            for packet in self.astream.encode():
                self.container.mux(packet)
        self.container.close()


def write_mp4_chunks(path: str, chunks, *, width: int, height: int, fps: int = G.FPS,
                     audio: Optional[np.ndarray] = None, sample_rate: int = G.AUDIO_SAMPLE_RATE,
                     overlap: bool = False, lookahead: int = 12) -> int:
    """Write ``uint8 [N, H, W, 3]`` blocks from an iterable; returns the frame count.

    ``overlap=True`` encodes on a dedicated worker thread while the caller keeps
    producing blocks, so x264 (2.6 s at 480p, 11 s at 1080p) runs during the VAE
    decode instead of after it. Frames, encoder and muxer stay strictly
    single-threaded in the same order, so the produced file is identical either
    way (verified byte-for-byte); only the wall clock changes. ``lookahead``
    bounds how many decoded frames may wait for the encoder, which caps memory.
    """
    if not overlap:
        writer = _Mp4Writer(path, width, height, fps=fps, audio=audio, sample_rate=sample_rate)
        try:
            for chunk in chunks:
                writer.write(chunk)
        finally:
            writer.close()
        return writer.frames

    import queue
    import threading

    frames_written = [0]
    failure: list[BaseException] = []

    def encode_all(pending: "queue.Queue") -> None:
        writer = _Mp4Writer(path, width, height, fps=fps, audio=audio, sample_rate=sample_rate)
        try:
            while True:
                chunk = pending.get()
                if chunk is None:
                    break
                writer.write(chunk)
            frames_written[0] = writer.frames
        except BaseException as exc:  # noqa: BLE001 - re-raised on the producing thread
            failure.append(exc)
        finally:
            try:
                writer.close()
            except BaseException as exc:  # noqa: BLE001
                failure.append(exc)

    pending: "queue.Queue" = queue.Queue(maxsize=max(1, int(lookahead)))
    encoder = threading.Thread(target=encode_all, args=(pending,), name="pika-vae-encode", daemon=True)
    encoder.start()
    try:
        for chunk in chunks:
            pending.put(chunk)
    finally:
        pending.put(None)
        encoder.join()
    if failure:
        raise failure[0]
    return frames_written[0]


def write_mp4(path: str, frames: np.ndarray, fps: int = G.FPS, audio: Optional[np.ndarray] = None, sample_rate: int = G.AUDIO_SAMPLE_RATE) -> None:
    """Write frames (+ optional stereo audio) with PyAV."""
    write_mp4_chunks(path, (frames,), width=frames.shape[2], height=frames.shape[1],
                     fps=fps, audio=audio, sample_rate=sample_rate)
