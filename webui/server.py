# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Pika H3 Flex browser WebUI: a standard-library HTTP server and task runner.

The server validates a request, stores uploads under one unique run directory,
queues the job (one accelerator, one generation at a time) and supervises
``python -m pika_h3.cli`` as a subprocess. Progress comes from ``[progress]``
markers printed by :mod:`pika_h3.pipeline`; every line is also streamed to the
browser and kept in ``<run>/task.log``.

Run from the repository root::

    python -m webui.server

Environment:
    HOST / PORT             bind address (default ``127.0.0.1:3000``)
    PIKA_WEBUI_DRY_RUN=1    exercise the API without loading models
    PIKA_WEBUI_RUNS_DIR     run directory (default ``webui/runs``)
    PIKA_WEBUI_BUDGET       default compute budget in MP·frames (default 260)
    PIKA_WEBUI_NO_DEVICE    set to 1 to skip ``npu-smi`` probing
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from webui import geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBUI_ROOT = REPO_ROOT / "webui"
STATIC_ROOT = WEBUI_ROOT / "static"
RUNS_DIR = Path(os.environ.get("PIKA_WEBUI_RUNS_DIR", str(WEBUI_ROOT / "runs"))).resolve()
RELEASE_FILE = REPO_ROOT / "release.json"

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 256 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_TOTAL_REQUEST_BYTES = 512 * 1024 * 1024
MAX_PROMPT_LENGTH = 8000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCE_TOTAL = 12
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 15.0
MAX_TOTAL_CLIP_SECONDS = 15.0
MAX_SEED = 2**63 - 1

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
STATIC_FILES = {
    "index.html", "logs.html", "style.css", "app.js", "logs.js", "seed.js", "favicon.svg",
}
TASK_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
ACTIVE_STATUSES = {"queued", "running"}
MODE_LABELS = {"text": "文本", "keyframes": "关键帧", "firstlast": "首尾帧", "references": "多模态参考"}
TASK_LABELS = {"t2va": "文本生成", "i2va": "首帧", "l2va": "尾帧", "fl2va": "首尾帧", "ref2va": "参考生成"}
STAGE_LABELS = {
    "queued": "排队等待", "prepare": "准备", "loading": "加载模型",
    "conditioning": "条件编码", "sampling": "采样去噪", "decode": "解码音视频",
    "encode": "封装 MP4", "done": "完成", "error": "失败", "canceled": "已取消",
}
PROGRESS_PATTERN = re.compile(r"^\[progress\]\s+(.+)$")

DEFAULT_BUDGET = float(os.environ.get("PIKA_WEBUI_BUDGET", "2000") or 2000)


class SubmissionError(ValueError):
    """A user-controlled request failed validation."""


class TaskConflictError(RuntimeError):
    """The server cannot accept a task right now."""


@dataclass(frozen=True)
class UploadedFile:
    field: str
    filename: str
    content: bytes
    content_type: str = ""


@dataclass(frozen=True)
class Submission:
    mode: str
    task: str
    prompt: str
    plan: dict[str, Any]
    seed: int
    low_vram: bool
    tensor_parallel: bool
    uploads: dict[str, list[UploadedFile]]
    model_repo: str
    revision: str
    base_path: str
    adapter_path: str
    text_encoder_path: str


@dataclass
class Task:
    id: str
    mode: str
    task: str
    prompt: str
    plan: dict[str, Any]
    seed: int
    low_vram: bool
    tensor_parallel: bool
    model: dict[str, str]
    input_files: list[str] = field(default_factory=list)
    status: str = "queued"
    stage: str = "queued"
    stage_label: str = "排队等待"
    progress: int = 0
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    cancel_requested: bool = False
    queue_position: int | None = 1
    log_bytes: int = 0
    output_ready: bool = False
    metadata_ready: bool = False
    dry_run: bool = False
    process_started_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        elapsed = self.elapsed_seconds
        if self.status == "running" and self.process_started_at is not None:
            elapsed = time.monotonic() - self.process_started_at
        data = {
            "id": self.id, "mode": self.mode, "mode_label": MODE_LABELS.get(self.mode, self.mode),
            "task": self.task, "task_label": TASK_LABELS.get(self.task, self.task),
            "status": self.status, "stage": self.stage, "stage_label": self.stage_label,
            "progress": self.progress, "seed": self.seed,
            "plan": self.plan, "width": self.plan.get("width"),
            "height": self.plan.get("height"), "frames": self.plan.get("frames"),
            "duration_seconds": self.plan.get("duration_seconds"),
            "seconds": self.plan.get("seconds"),
            "low_vram": self.low_vram, "tensor_parallel": self.tensor_parallel,
            "model": self.model, "input_files": self.input_files,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "elapsed_seconds": round(elapsed, 2),
            "error": self.error, "queue_position": self.queue_position,
            "log_bytes": self.log_bytes, "output_ready": self.output_ready,
            "metadata_ready": self.metadata_ready, "dry_run": self.dry_run,
            "prompt_chars": len(self.prompt),
        }
        data.update(self.extra)
        return data


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _one(fields: Mapping[str, list[str]], name: str, default: str = "") -> str:
    values = fields.get(name, [])
    if len(values) > 1:
        raise SubmissionError(f"{name} 只能提交一次")
    return values[0] if values else default


def _parse_int(value: str, name: str, minimum: int, maximum: int) -> int:
    if not re.fullmatch(r"-?[0-9]+", (value or "").strip()):
        raise SubmissionError(f"{name} 必须是整数")
    number = int(value)
    if not minimum <= number <= maximum:
        raise SubmissionError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return number


def _parse_seed(value: str | None) -> int:
    if value is None or not str(value).strip():
        return random_seed()
    return _parse_int(str(value), "seed", 0, MAX_SEED)


def random_seed() -> int:
    return int.from_bytes(os.urandom(8), "big") & MAX_SEED


def _parse_checkbox(fields: Mapping[str, list[str]], name: str, default: bool) -> bool:
    value = _one(fields, name, "true" if default else "false").strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise SubmissionError(f"{name} 必须为 true 或 false")
    return value in {"1", "true", "yes", "on"}


def _safe_filename(filename: str) -> str:
    """Single safe path component, retaining ordinary names (spaces, Unicode)."""
    filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    filename = re.sub(r"[\x00-\x1f\x7f]", "_", filename)
    filename = re.sub(r'[<>:"|?*]', "_", filename).strip()
    if filename in {"", ".", ".."}:
        raise SubmissionError("上传文件需要有文件名")
    if len(filename.encode("utf-8")) > 180:
        suffix = Path(filename).suffix
        if len(suffix) > 10:
            suffix = ""
        stem = filename[: -len(suffix)] if suffix else filename
        stem = stem.encode("utf-8")[: 180 - len(suffix.encode("utf-8"))].decode("utf-8", errors="ignore")
        filename = stem + suffix
    return filename


def _check_image(data: bytes, suffix: str, name: str) -> None:
    valid = {
        ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": data.startswith(b"\xff\xd8\xff"),
        ".jpeg": data.startswith(b"\xff\xd8\xff"),
        ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
        ".bmp": data.startswith(b"BM"),
    }.get(suffix, False)
    if not valid:
        raise SubmissionError(f"{name} 不是有效的图片内容")


def _check_video(data: bytes, name: str) -> None:
    valid = (data[4:8] == b"ftyp" or data.startswith(b"\x1aE\xdf\xa3")
             or (data.startswith(b"RIFF") and data[8:12] == b"AVI "))
    if not valid:
        raise SubmissionError(f"{name} 不是有效的视频内容")


def _check_audio(data: bytes, name: str) -> None:
    valid = (data.startswith((b"RIFF", b"ID3", b"fLaC", b"OggS"))
             or data[4:8] == b"ftyp"
             or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xF0 == 0xF0))
    if not valid:
        raise SubmissionError(f"{name} 不是有效的音频内容")


def _validate_uploads(uploads: Mapping[str, list[UploadedFile]]) -> None:
    for field, files in uploads.items():
        for uploaded in files:
            safe_name = _safe_filename(uploaded.filename)
            suffix = Path(safe_name).suffix.lower()
            data = uploaded.content
            if not data:
                raise SubmissionError(f"上传文件 {uploaded.filename!r} 为空")
            if field in {"first", "last", "refs"}:
                if suffix not in ALLOWED_IMAGE_SUFFIXES:
                    raise SubmissionError(
                        f"{uploaded.filename!r} 不是受支持的图片（{', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}）")
                if len(data) > MAX_IMAGE_BYTES:
                    raise SubmissionError(f"图片 {uploaded.filename!r} 超过 20 MiB 限制")
                _check_image(data, suffix, uploaded.filename)
            elif field == "video_refs":
                if suffix not in ALLOWED_VIDEO_SUFFIXES:
                    raise SubmissionError(
                        f"{uploaded.filename!r} 不是受支持的视频（{', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}）")
                if len(data) > MAX_VIDEO_BYTES:
                    raise SubmissionError(f"视频 {uploaded.filename!r} 超过 256 MiB 限制")
                _check_video(data, uploaded.filename)
            elif field == "audio_refs":
                if suffix not in ALLOWED_AUDIO_SUFFIXES:
                    raise SubmissionError(
                        f"{uploaded.filename!r} 不是受支持的音频（{', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}）")
                if len(data) > MAX_AUDIO_BYTES:
                    raise SubmissionError(f"音频 {uploaded.filename!r} 超过 64 MiB 限制")
                _check_audio(data, uploaded.filename)
            else:
                raise SubmissionError(f"未知上传字段 {field!r}")


def probe_media_duration(path: Path, kind: str) -> float:
    """Container duration in seconds via ffprobe; PyAV fallback; reject if unknown."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        value = result.stdout.strip().splitlines()
        if result.returncode == 0 and value:
            duration = float(value[-1])
            if duration > 0:
                return duration
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import av  # type: ignore

        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            streams = container.streams.video if kind == "video" else container.streams.audio
            stream = streams[0]
            if stream.duration is not None and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:
        pass
    raise SubmissionError(f"无法读取{('视频' if kind == 'video' else '音频')}时长：{path.name}")


def validate_reference_media(uploads: Mapping[str, list[UploadedFile]], tmp_dir: Path | None = None) -> dict[str, Any]:
    """Count and duration rules from README "Generation modes and reference limits"."""
    images = len(uploads.get("refs", []))
    videos = uploads.get("video_refs", [])
    audios = uploads.get("audio_refs", [])
    if images > MAX_REFERENCE_IMAGES:
        raise SubmissionError(f"ref2va 最多 {MAX_REFERENCE_IMAGES} 张参考图片（收到 {images} 张）")
    if len(videos) > MAX_REFERENCE_VIDEOS:
        raise SubmissionError(f"ref2va 最多 {MAX_REFERENCE_VIDEOS} 个参考视频（收到 {len(videos)} 个）")
    if len(audios) > MAX_REFERENCE_AUDIOS:
        raise SubmissionError(f"ref2va 最多 {MAX_REFERENCE_AUDIOS} 个独立参考音频（收到 {len(audios)} 个）")
    total = images + len(videos) + len(audios)
    if total > MAX_REFERENCE_TOTAL:
        raise SubmissionError(f"参考文件总计最多 {MAX_REFERENCE_TOTAL} 个（收到 {total} 个）")
    if images + len(videos) == 0:
        raise SubmissionError("ref2va 至少需要一张参考图片或一个参考视频")
    durations = {"video": [], "audio": []}
    scratch = None
    try:
        for kind, files in (("video", videos), ("audio", audios)):
            for uploaded in files:
                if tmp_dir is not None:
                    probe = Path(tmp_dir) / f".probe-{uuid.uuid4().hex}{Path(uploaded.filename).suffix.lower()}"
                    probe.write_bytes(uploaded.content)
                else:
                    scratch = scratch or Path(os.environ.get("TMPDIR", "/tmp")) / f"pika-probe-{uuid.uuid4().hex}"
                    scratch.mkdir(parents=True, exist_ok=True)
                    probe = scratch / _safe_filename(uploaded.filename)
                    probe.write_bytes(uploaded.content)
                try:
                    duration = probe_media_duration(probe, kind)
                finally:
                    try:
                        probe.unlink()
                    except OSError:
                        pass
                label = "视频" if kind == "video" else "音频"
                if not MIN_CLIP_SECONDS - 1e-6 <= duration <= MAX_CLIP_SECONDS + 1e-6:
                    raise SubmissionError(
                        f"{label} {uploaded.filename!r} 时长 {duration:.2f}s，必须为 2–15 秒")
                durations[kind].append(round(duration, 3))
    finally:
        if scratch is not None:
            try:
                for leftover in scratch.iterdir():
                    leftover.unlink()
                scratch.rmdir()
            except OSError:
                pass
    video_total = sum(durations["video"])
    audio_total = sum(durations["audio"])
    if video_total > MAX_TOTAL_CLIP_SECONDS + 1e-6:
        raise SubmissionError(f"参考视频总时长 {video_total:.2f}s 超过 15 秒上限")
    if audio_total > MAX_TOTAL_CLIP_SECONDS + 1e-6:
        raise SubmissionError(f"独立参考音频总时长 {audio_total:.2f}s 超过 15 秒上限")
    return {
        "counts": {"image": images, "video": len(videos), "audio": len(audios), "total": total},
        "durations": durations,
        "video_seconds": round(video_total, 3),
        "audio_seconds": round(audio_total, 3),
    }


def _load_release() -> dict[str, Any]:
    try:
        return json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _validate_model_path(value: str, name: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if "\x00" in value or len(value) > 1024:
        raise SubmissionError(f"{name} 路径无效")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def compute_plan_from_fields(fields: Mapping[str, list[str]]) -> dict[str, Any]:
    """Server-authoritative canvas/duration decision (MP mode or preset mode)."""
    canvas_mode = _one(fields, "canvas_mode", "mp").strip().lower()
    if canvas_mode not in {"mp", "preset"}:
        raise SubmissionError("canvas_mode 必须为 mp 或 preset")
    aspect = _one(fields, "aspect", "16:9").strip()
    if aspect not in geometry.ASPECTS:
        raise SubmissionError(f"不支持的宽高比 {aspect!r}")
    budget_text = _one(fields, "budget", str(DEFAULT_BUDGET)).strip()
    try:
        budget = float(budget_text)
    except ValueError:
        raise SubmissionError("budget 必须是数字") from None
    if not (geometry.MIN_BUDGET <= budget <= geometry.MAX_BUDGET):
        raise SubmissionError(
            f"预算必须在 {geometry.MIN_BUDGET:g}–{geometry.MAX_BUDGET:g} MP·帧 之间")
    override = _parse_checkbox(fields, "override", False)
    override_canvas = None
    override_duration = None
    if override:
        width_text = _one(fields, "override_width", "").strip()
        height_text = _one(fields, "override_height", "").strip()
        if width_text or height_text:
            width = _parse_int(width_text, "override_width", 32, geometry.MAX_CANVAS_SIDE)
            height = _parse_int(height_text, "override_height", 32, geometry.MAX_CANVAS_SIDE)
            try:
                geometry.validate_canvas(width, height)
            except ValueError as exc:
                raise SubmissionError(str(exc)) from None
            override_canvas = (width, height)
        duration_text = _one(fields, "override_duration", "").strip()
        if duration_text:
            override_duration = _parse_int(
                duration_text, "override_duration", geometry.MIN_SECONDS, geometry.MAX_SECONDS)
    if canvas_mode == "preset":
        resolution = _one(fields, "resolution", "768p").strip()
        if resolution not in geometry.RESOLUTIONS:
            raise SubmissionError(f"resolution 必须为 {', '.join(geometry.RESOLUTIONS)}")
        duration_text = _one(fields, "duration", "").strip()
        if not duration_text:
            if override_duration is None:
                raise SubmissionError("预设模式需要时长")
            duration_text = str(override_duration)
        duration = _parse_int(duration_text, "duration", geometry.MIN_SECONDS, geometry.MAX_SECONDS)
        if override_canvas is not None:
            # Preset mode still supports a forced smaller canvas for memory relief.
            plan = geometry.plan_canvas(
                mp=override_canvas[0] * override_canvas[1] / 1e6, aspect=aspect, budget=budget,
                canvas=override_canvas, duration=duration)
            plan["resolution"] = resolution
            plan["canvas_mode"] = "preset"
        else:
            plan = geometry.preset_plan(resolution, aspect, duration, budget=budget)
        return plan
    mp_text = _one(fields, "mp", "1.0").strip()
    try:
        mp = float(mp_text)
    except ValueError:
        raise SubmissionError("mp 必须是数字（单位：百万像素）") from None
    if not (geometry.MIN_MP <= mp <= geometry.MAX_MP):
        raise SubmissionError(f"MP 必须在 {geometry.MIN_MP:g}–{geometry.MAX_MP:g} 之间")
    # Duration is a user choice in MP mode too; the budget only warns.
    duration_text = _one(fields, "duration", "").strip()
    duration = override_duration
    if duration_text:
        duration = _parse_int(duration_text, "duration", geometry.MIN_SECONDS, geometry.MAX_SECONDS)
    if duration is None:
        raise SubmissionError("MP 模式需要时长")
    # Canvas shrinking is opt-in: by default an over-budget MP/aspect is rejected
    # instead of being silently replaced by a smaller canvas.
    allow_shrink = _one(fields, "allow_canvas_shrink", "").strip().lower() in ("1", "true", "on", "yes")
    return geometry.plan_canvas(mp=mp, aspect=aspect, budget=budget,
                                canvas=override_canvas, duration=duration,
                                allow_shrink=allow_shrink)


def validate_submission(fields: Mapping[str, list[str]], uploads: Mapping[str, list[UploadedFile]]) -> tuple[Submission, dict[str, Any]]:
    """Validate form data and return a command-safe request description."""
    mode = _one(fields, "mode", "text").strip().lower()
    if mode == "first+last":
        mode = "firstlast"
    if mode in {"t2va", "i2va", "l2va", "fl2va", "ref2va"}:
        # Backwards-compatible raw task names map onto the four UI cards.
        mode = {"t2va": "text", "i2va": "keyframes", "l2va": "keyframes",
                "fl2va": "firstlast", "ref2va": "references"}[mode]
    if mode not in MODE_LABELS:
        raise SubmissionError("mode 必须为 text、keyframes、firstlast（first+last）或 references")

    prompt = _one(fields, "prompt")
    if "\x00" in prompt or not prompt.strip():
        raise SubmissionError("提示词不能为空")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise SubmissionError(f"提示词最多 {MAX_PROMPT_LENGTH} 个字符")

    normalized_uploads: dict[str, list[UploadedFile]] = {
        "first": list(uploads.get("first", [])),
        "last": list(uploads.get("last", [])),
        "refs": list(uploads.get("refs", [])) + list(uploads.get("refs[]", [])),
        "video_refs": list(uploads.get("video_refs", [])),
        "audio_refs": list(uploads.get("audio_refs", [])),
    }
    if any(name not in {"first", "last", "refs", "refs[]", "video_refs", "audio_refs"} for name in uploads):
        raise SubmissionError("未知上传字段")
    _validate_uploads(normalized_uploads)

    if mode == "text":
        task = "t2va"
        if any(normalized_uploads.values()):
            raise SubmissionError("文本模式不接受图片或参考素材")
    elif mode == "keyframes":
        first = normalized_uploads["first"]
        last = normalized_uploads["last"]
        if any(normalized_uploads[name] for name in ("refs", "video_refs", "audio_refs")):
            raise SubmissionError("关键帧模式不接受参考素材")
        if len(first) > 1 or len(last) > 1:
            raise SubmissionError("首帧和尾帧各只能有一张图片")
        if not first and not last:
            raise SubmissionError("关键帧模式需要上传首帧、尾帧或两者")
        task = "i2va" if first and not last else "l2va" if last and not first else "fl2va"
    elif mode == "firstlast":
        task = "fl2va"
        if len(normalized_uploads["first"]) != 1 or len(normalized_uploads["last"]) != 1:
            raise SubmissionError("首尾帧模式需要同时上传一张首帧和一张尾帧")
        if any(normalized_uploads[name] for name in ("refs", "video_refs", "audio_refs")):
            raise SubmissionError("首尾帧模式不接受参考素材")
    else:
        task = "ref2va"
        if normalized_uploads["first"] or normalized_uploads["last"]:
            raise SubmissionError("参考模式请使用参考图片/视频/音频字段")
    media = (validate_reference_media(normalized_uploads) if mode == "references"
             else {"counts": {"image": 0, "video": 0, "audio": 0, "total": 0},
                   "durations": {"video": [], "audio": []}, "video_seconds": 0, "audio_seconds": 0})

    plan = compute_plan_from_fields(fields)
    seed = _parse_seed(_one(fields, "seed", "") or None)
    low_vram = _parse_checkbox(fields, "low_vram", True)
    tensor_parallel = _parse_checkbox(fields, "tensor_parallel", False)
    release = _load_release()
    model_repo = _one(fields, "model_repo", "Pika-Labs/Pika-H3-Flex").strip() or "Pika-Labs/Pika-H3-Flex"
    revision = _one(fields, "revision", str(release.get("model_revision", ""))).strip()
    base_path = _validate_model_path(
        _one(fields, "base_path", os.environ.get("PIKA_BASE_PATH", "models/MiniMax-H3-ref")), "base_path")
    adapter_path = _validate_model_path(
        _one(fields, "adapter_path", os.environ.get("PIKA_ADAPTER_PATH", "models/Pika-H3-Flex")), "adapter_path")
    text_encoder_path = _validate_model_path(
        _one(fields, "text_encoder_path", os.environ.get("PIKA_TEXT_ENCODER_PATH", "")), "text_encoder_path")
    if not adapter_path and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SubmissionError("未提供本地 adapter 路径时，revision 必须是 40 位提交 SHA")
    for label, value in (("base_path", base_path), ("adapter_path", adapter_path),
                         ("text_encoder_path", text_encoder_path)):
        if value and not Path(value).is_dir():
            raise SubmissionError(f"{label} 目录不存在：{value}")
    if base_path:
        expected = Path(base_path) / ("transformer_ref" if task == "ref2va" else "transformer")
        if not expected.is_dir():
            raise SubmissionError(f"base_path 缺少 {expected.name} 目录：{base_path}")
    if adapter_path:
        branch = "ref2va" if task == "ref2va" else "fl2va"
        expected = Path(adapter_path) / branch / "adapter.safetensors"
        if not expected.is_file():
            raise SubmissionError(f"adapter_path 缺少 {branch}/adapter.safetensors：{adapter_path}")

    submission = Submission(
        mode=mode, task=task, prompt=prompt, plan=plan, seed=seed,
        low_vram=low_vram, tensor_parallel=tensor_parallel, uploads=normalized_uploads,
        model_repo=model_repo, revision=revision, base_path=base_path,
        adapter_path=adapter_path, text_encoder_path=text_encoder_path,
    )
    return submission, media


def build_command(task: Task, run_dir: Path) -> list[str]:
    """Build the ``python -m pika_h3.cli`` argv for a validated task."""
    python = sys.executable
    cli_args = [
        "--task", task.task,
        "--prompt-file", str(run_dir / "prompt.txt"),
        "--canvas", f"{task.plan['width']}x{task.plan['height']}",
        "--duration", str(int(task.plan["duration_seconds"])),
        "--seed", str(task.seed),
        "--output", str(run_dir / "output.mp4"),
        "--model-repo", task.model["repo"],
        "--revision", task.model["revision"],
    ]
    if task.model.get("base_path"):
        cli_args += ["--base-path", task.model["base_path"]]
    if task.model.get("adapter_path"):
        # ``--adapter-path`` is the per-task adapter directory (…/fl2va or
        # …/ref2va), not the repository root that holds both branches.
        branch = "ref2va" if task.task == "ref2va" else "fl2va"
        cli_args += ["--adapter-path", str(Path(task.model["adapter_path"]) / branch)]
    if task.model.get("text_encoder_path"):
        cli_args += ["--text-encoder-path", task.model["text_encoder_path"]]
    if task.low_vram:
        cli_args.append("--low-vram")
    if task.tensor_parallel:
        cli_args += ["--tensor-parallel", "2"]
    inputs = run_dir / "inputs"
    first = sorted(inputs.glob("first.*"))
    last = sorted(inputs.glob("last.*"))
    images = sorted(inputs.glob("image-*"))
    videos = sorted(inputs.glob("video-*"))
    audios = sorted(inputs.glob("audio-*"))
    if first:
        cli_args += ["--first-image", str(first[0])]
    if last:
        cli_args += ["--last-image", str(last[0])]
    # Reference order is preserved and grouped by modality, matching
    # <Picture n>, <Video n> then <Audio n> in the prompt.
    for kind, paths in (("image", images), ("video", videos), ("audio", audios)):
        for path in paths:
            if path.is_file():
                cli_args += ["--ref", f"{kind}:{path}"]
    command = [python, "-u", "-m", "pika_h3.cli"] + cli_args
    if task.tensor_parallel:
        command = [python, "-u", "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=2", "-m", "pika_h3.cli"] + cli_args
    return command


class TaskManager:
    """Own run directories, the FIFO queue, one subprocess, and state/history."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._tasks: dict[str, Task] = {}
        self._pending: deque[str] = deque()
        self._active_id: str | None = None
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._worker = threading.Thread(target=self._worker_loop, name="pika-webui-worker", daemon=True)
        self._worker.start()
        self.dry_run = os.environ.get("PIKA_WEBUI_DRY_RUN", "").strip() == "1"
        try:
            self.dry_run_delay = max(0.0, float(os.environ.get("PIKA_WEBUI_DRY_RUN_DELAY", "0.35")))
        except ValueError:
            self.dry_run_delay = 0.35
        self._device_cache: dict[str, Any] = {"at": 0.0, "value": None}
        self._load_history()

    # ---- storage helpers -------------------------------------------------------------------------------------
    def _run_dir(self, task_id: str) -> Path:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise KeyError(task_id)
        path = self.runs_dir / task_id
        if path.resolve() != path:
            raise KeyError(task_id)
        return path

    def _artifact_path(self, task_id: str, filename: str) -> Path:
        run_dir = self._run_dir(task_id)
        path = run_dir / filename
        if not path.resolve().is_relative_to(run_dir):
            raise KeyError(task_id)
        return path

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _persist(self, task: Task) -> None:
        public = task.to_public()
        public["prompt"] = task.prompt
        self._atomic_write(self._run_dir(task.id) / "state.json", public)

    def _load_history(self) -> None:
        for state_path in sorted(self.runs_dir.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    continue
                task_id = str(state.get("id", ""))
                if not TASK_ID_PATTERN.fullmatch(task_id) or state_path.parent.name != task_id:
                    continue
                plan = state.get("plan") or {}
                task = Task(
                    id=task_id, mode=str(state.get("mode", "text")),
                    task=str(state.get("task", "t2va")), prompt=str(state.get("prompt", "")),
                    plan=plan, seed=int(state.get("seed", 0)),
                    low_vram=bool(state.get("low_vram", True)),
                    tensor_parallel=bool(state.get("tensor_parallel", False)),
                    model=state.get("model") or {},
                    input_files=list(state.get("input_files") or []),
                    status=str(state.get("status", "error")),
                    stage=str(state.get("stage", "error")),
                    stage_label=str(state.get("stage_label", "失败")),
                    progress=int(state.get("progress", 0)),
                    created_at=str(state.get("created_at", "")),
                    started_at=state.get("started_at"), finished_at=state.get("finished_at"),
                    elapsed_seconds=float(state.get("elapsed_seconds", 0) or 0),
                    error=state.get("error"), dry_run=bool(state.get("dry_run", False)),
                    output_ready=bool(state.get("output_ready", False)),
                    metadata_ready=bool(state.get("metadata_ready", False)),
                    queue_position=None,
                )
                if task.status in ACTIVE_STATUSES:
                    task.status, task.stage, task.stage_label = "error", "error", "失败"
                    task.error = "服务重启时任务仍在运行，已标记为中断"
                    task.queue_position = None
                    self._persist(task)
                self._tasks[task_id] = task
            except (OSError, ValueError, TypeError, KeyError):
                continue

    # ---- public views ----------------------------------------------------------------------------------------
    def _queue_positions(self) -> dict[str, int]:
        return {task_id: index + 1 for index, task_id in enumerate(self._pending)}

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            self._refresh_queue(task)
            return task.to_public()

    def task_object(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            positions = self._queue_positions()
            tasks = []
            for index, task in enumerate(self._tasks.values()):
                task.queue_position = positions.get(task.id)
                tasks.append((index, task.to_public()))
        # Same-second tasks keep submission order: newest insertion first.
        tasks.sort(key=lambda item: (str(item[1].get("created_at", "")), item[0]), reverse=True)
        return [task for _, task in tasks]

    def _refresh_queue(self, task: Task) -> None:
        if task.id in self._pending:
            task.queue_position = list(self._pending).index(task.id) + 1
        elif task.id != self._active_id:
            task.queue_position = None

    def read_log(self, task_id: str, tail_bytes: int | None = 256 * 1024) -> str:
        path = self._artifact_path(task_id, "task.log")
        try:
            with path.open("rb") as log_file:
                if tail_bytes is not None and tail_bytes > 0:
                    log_file.seek(0, os.SEEK_END)
                    size = log_file.tell()
                    log_file.seek(max(0, size - tail_bytes))
                data = log_file.read()
            return data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def artifact(self, task_id: str, filename: str) -> Path | None:
        try:
            path = self._artifact_path(task_id, filename)
        except KeyError:
            return None
        return path if path.is_file() else None

    def queue_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "pending_ids": list(self._pending),
                "active": self._active_id,
            }

    # ---- submission ------------------------------------------------------------------------------------------
    def submit(self, submission: Submission, media: Mapping[str, Any]) -> str:
        with self._lock:
            if self._closed:
                raise TaskConflictError("服务正在停止")
            task_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = self._run_dir(task_id)
            input_dir = run_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "prompt.txt").write_text(submission.prompt, encoding="utf-8")
            input_files: list[str] = []
            counters = {"first": 0, "last": 0, "refs": 0, "video_refs": 0, "audio_refs": 0}
            for field in ("first", "last", "refs", "video_refs", "audio_refs"):
                for uploaded in submission.uploads.get(field, []):
                    counters[field] += 1
                    safe = _safe_filename(uploaded.filename)
                    suffix = Path(safe).suffix.lower()
                    prefix = {"first": "first", "last": "last", "refs": "image",
                              "video_refs": "video", "audio_refs": "audio"}[field]
                    name = f"{prefix}-{counters[field]}{suffix}"
                    if field in {"first", "last"}:
                        name = f"{prefix}{suffix}"
                    destination = input_dir / name
                    destination.write_bytes(uploaded.content)
                    input_files.append(str(destination.relative_to(run_dir)))
            plan = dict(submission.plan)
            task = Task(
                id=task_id, mode=submission.mode, task=submission.task,
                prompt=submission.prompt, plan=plan, seed=submission.seed,
                low_vram=submission.low_vram, tensor_parallel=submission.tensor_parallel,
                model={"repo": submission.model_repo, "revision": submission.revision,
                       "base_path": submission.base_path, "adapter_path": submission.adapter_path,
                       "text_encoder_path": submission.text_encoder_path},
                input_files=input_files, status="queued", stage="queued", stage_label="排队等待",
                created_at=utc_now(), dry_run=self.dry_run,
                extra={"reference_counts": media.get("counts"), "reference_durations": media.get("durations")},
            )
            task.queue_position = len(self._pending) + 1
            self._tasks[task_id] = task
            self._append_log(task_id, f"[webui] 任务已创建 {task_id}")
            self._append_log(task_id, f"[webui] 模式={submission.mode} 任务={submission.task} "
                                       f"种子={submission.seed} 画布={plan['width']}x{plan['height']} "
                                       f"帧={plan['frames']} 时长={plan['duration_seconds']}s")
            self._persist(task)
            self._pending.append(task_id)
            self._condition.notify_all()
            return task_id

    # ---- logging & progress ---------------------------------------------------------------------------------
    def _append_log(self, task_id: str, line: str) -> None:
        path = self._artifact_path(task_id, "task.log")
        line = line.rstrip("\n")
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
        task = self._tasks.get(task_id)
        if task is not None:
            task.log_bytes += len(line.encode("utf-8")) + 1

    def _set_stage(self, task_id: str, stage: str, progress: int, message: str | None = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "running"
            task.stage = stage
            task.stage_label = STAGE_LABELS.get(stage, stage)
            task.progress = max(task.progress, max(0, min(100, int(progress))))
            if task.started_at is None:
                task.started_at = utc_now()
                task.process_started_at = time.monotonic()
            self._persist(task)
        if message:
            self._append_log(task_id, f"[webui] {message}")

    def _record_line(self, task_id: str, raw: str) -> None:
        line = raw.rstrip("\r\n")
        if not line:
            return
        self._append_log(task_id, line)
        match = PROGRESS_PATTERN.match(line.strip())
        if match:
            data: dict[str, str] = {}
            for token in match.group(1).split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    data[key] = value
            stage = data.get("stage", "")
            try:
                pct = int(float(data.get("pct", "0")))
            except ValueError:
                return
            if stage in STAGE_LABELS:
                # ``done`` is emitted by the pipeline before the JSON is written and the
                # process exits; keep 100 % and the "done" state for the manager's own
                # finalization so the task cannot look finished while the child runs.
                if stage == "done":
                    self._set_stage(task_id, "encode", 99)
                else:
                    self._set_stage(task_id, stage, pct)
            return
        lowered = line.lower()
        # Fallbacks for logs emitted before/without the markers.
        if "[cond] qwen3-vl loaded" in lowered or "loading checkpoint shards" in lowered:
            self._set_stage(task_id, "loading", max(self._progress(task_id), 24))
        elif "[cond] vaes loaded" in lowered:
            self._set_stage(task_id, "conditioning", max(self._progress(task_id), 32))
        elif "saved " in lowered and ".mp4" in lowered:
            self._set_stage(task_id, "encode", 98)

    def _progress(self, task_id: str) -> int:
        task = self._tasks.get(task_id)
        return int(task.progress) if task else 0

    # ---- worker ----------------------------------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait(timeout=0.5)
                if self._closed and not self._pending:
                    return
                task_id = self._pending.popleft()
                self._active_id = task_id
                task = self._tasks.get(task_id)
                if task is None or task.status == "canceled":
                    self._active_id = None
                    continue
                task.queue_position = None
            try:
                self._run_task(task_id)
            except Exception as exc:  # noqa: BLE001 - surfaced as task state
                self._fail(task_id, str(exc) or exc.__class__.__name__)
            finally:
                with self._lock:
                    if self._active_id == task_id:
                        self._active_id = None
                    self._process = None

    def _fail(self, task_id: str, message: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            cancelled = task.cancel_requested
            if cancelled:
                task.status, task.stage, task.stage_label = "canceled", "canceled", "已取消"
                task.error = "任务已取消"
            else:
                task.status, task.stage, task.stage_label = "error", "error", "失败"
                task.error = message
            task.finished_at = utc_now()
            task.queue_position = None
            if task.process_started_at is not None:
                task.elapsed_seconds = time.monotonic() - task.process_started_at
            self._persist(task)
        self._append_log(task_id, f"[webui] {'已取消' if cancelled else '错误'}: {message}")

    def _run_task(self, task_id: str) -> None:
        run_dir = self._run_dir(task_id)
        with self._lock:
            task = self._tasks[task_id]
            cancelled_before_start = task.cancel_requested
        if cancelled_before_start:
            self._fail(task_id, "任务已取消")
            return
        task.process_started_at = time.monotonic()
        command = build_command(task, run_dir)
        self._append_log(task_id, "[webui] 执行: " + shlex.join(command))
        try:
            if self.dry_run:
                self._run_dry(task_id, run_dir, command)
            else:
                self._execute(task_id, command, run_dir)
        except Exception as exc:
            self._fail(task_id, str(exc) or exc.__class__.__name__)
            return

        with self._lock:
            task = self._tasks[task_id]
            cancelled = task.cancel_requested
        output = run_dir / "output.mp4"
        metadata = run_dir / "output.json"
        with self._lock:
            task = self._tasks[task_id]
            task.output_ready = output.is_file() and output.stat().st_size > 0
            task.metadata_ready = metadata.is_file()
            task.finished_at = utc_now()
            if task.process_started_at is not None:
                task.elapsed_seconds = time.monotonic() - task.process_started_at
            if cancelled:
                task.status, task.stage, task.stage_label = "canceled", "canceled", "已取消"
                task.error = "任务已取消"
            elif not task.output_ready:
                task.status, task.stage, task.stage_label = "error", "error", "失败"
                task.error = "生成结束但没有输出 MP4"
            else:
                task.status, task.stage, task.stage_label = "done", "done", "完成"
                task.progress = 100
                task.error = None
            task.queue_position = None
            self._persist(task)
        if not cancelled:
            self._append_log(task_id, f"[webui] 结束: {run_dir / 'output.mp4'}")

    def _execute(self, task_id: str, command: list[str], run_dir: Path) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        existing = environment.get("PYTHONPATH", "")
        source = str(REPO_ROOT / "src")
        environment["PYTHONPATH"] = source + (os.pathsep + existing if existing else "")
        environment.setdefault("HF_HUB_DISABLE_XET", "1")
        with self._lock:
            if self._closed or self._tasks[task_id].cancel_requested:
                raise RuntimeError("任务在启动前被取消")
            try:
                # The child gets its own session so cancellation can kill the whole
                # torchrun/worker process group, not just the direct child.
                process = subprocess.Popen(
                    command, cwd=str(REPO_ROOT), env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeError(f"无法启动子进程：{exc}") from exc
            self._process = process
        self._set_stage(task_id, "prepare", 2, "子进程已启动")
        assert process.stdout is not None
        reader_errors: list[Exception] = []

        def read_lines() -> None:
            try:
                with process.stdout:
                    for line in process.stdout:
                        self._record_line(task_id, line)
            except Exception as exc:  # noqa: BLE001
                reader_errors.append(exc)

        reader = threading.Thread(target=read_lines, name=f"pika-webui-log-{task_id}", daemon=True)
        reader.start()
        # Watch for cancellation while waiting; a plain wait() would only notice
        # after the (potentially minutes-long) generation finished.
        while True:
            try:
                return_code = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                with self._lock:
                    cancelled = self._tasks[task_id].cancel_requested
                if cancelled:
                    self._terminate_process_group(process)
                    try:
                        return_code = process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self._terminate_process_group(process, signal.SIGKILL)
                        return_code = process.wait()
                    break
        reader.join(timeout=30)
        with self._lock:
            self._process = None
            cancelled = self._tasks[task_id].cancel_requested
        if cancelled:
            raise RuntimeError("任务已取消")
        if reader_errors:
            raise RuntimeError(f"读取子进程输出失败：{reader_errors[0]}")
        if return_code != 0:
            raise RuntimeError(f"子进程退出码 {return_code}；详见任务日志")

    def _run_dry(self, task_id: str, run_dir: Path, command: list[str]) -> None:
        self._append_log(task_id, "[dry-run] 未加载模型；模拟进度与日志")
        plan = self._tasks[task_id].plan
        steps = [
            ("prepare", 2, "dry-run 准备"),
            ("loading", 8, "dry-run 资产校验"),
            ("loading", 20, "dry-run Qwen3-VL 加载"),
            ("conditioning", 40, "dry-run 条件编码完成"),
            ("loading", 45, "dry-run transformer 加载"),
            ("loading", 55, "dry-run transformer 加载完成"),
            ("sampling", 60, "dry-run 采样 0/6"),
            ("sampling", 70, "dry-run 采样 2/6"),
            ("sampling", 82, "dry-run 采样 6/6"),
            ("decode", 85, "dry-run 视频解码"),
            ("decode", 93, "dry-run 音频解码"),
            ("encode", 96, "dry-run 封装 MP4"),
            ("done", 99, "dry-run 完成"),
        ]
        for stage, pct, message in steps:
            with self._lock:
                task = self._tasks[task_id]
                task.process_started_at = task.process_started_at or time.monotonic()
                if task.cancel_requested:
                    raise RuntimeError("任务已取消")
            self._append_log(task_id, f"[dry-run] {message}")
            self._record_line(task_id, f"[progress] stage={stage} pct={pct} dry=1")
            time.sleep(self.dry_run_delay)
        self._append_log(task_id, f"[dry-run] 画布={plan['width']}x{plan['height']} 帧={plan['frames']} "
                                   f"时长={plan['duration_seconds']}s 种子={self._tasks[task_id].seed}")
        self._write_dry_outputs(run_dir, task_id)

    def _write_dry_outputs(self, run_dir: Path, task_id: str) -> None:
        task = self._tasks[task_id]
        output = run_dir / "output.mp4"
        made = False
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                 "color=c=0x101820:s=480x270:d=1:r=24",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output)],
                capture_output=True, timeout=30,
            )
            made = result.returncode == 0 and output.is_file() and output.stat().st_size > 0
        except (OSError, subprocess.SubprocessError):
            made = False
        if not made:
            output.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x08free")
        metadata = {
            "task": task.task, "seed": task.seed, "dry_run": True,
            "canvas": {"width": task.plan["width"], "height": task.plan["height"],
                       "num_frames": task.plan["frames"]},
            "geometry": {"height": task.plan["height"], "width": task.plan["width"],
                         "num_frames": task.plan["frames"]},
            "requested_seconds": task.plan["duration_seconds"],
            "plan": task.plan, "prompt_chars": len(task.prompt),
            "created_at": task.created_at, "webui": "dry-run",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        self._append_log(task_id, f"[dry-run] 写入 {output.name} 与 {output.with_suffix('.json').name}")

    # ---- cancellation & shutdown -----------------------------------------------------------------------------
    def cancel(self, task_id: str) -> str:
        """Cancel a queued task or terminate the running process group."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status not in ACTIVE_STATUSES:
                raise TaskConflictError("任务已结束，无法取消")
            task.cancel_requested = True
            if task_id in self._pending:
                self._pending.remove(task_id)
                task.status, task.stage, task.stage_label = "canceled", "canceled", "已取消"
                task.error = "任务已取消"
                task.finished_at = utc_now()
                task.queue_position = None
                self._persist(task)
                self._append_log(task_id, "[webui] 排队中的任务已取消")
                return "canceled-queued"
            process = self._process
        if process is not None and process.poll() is None:
            self._append_log(task_id, "[webui] 正在终止进程组…")
            self._terminate_process_group(process)
            return "terminating"
        # The worker is between dequeue and Popen; its pre-start check sees the flag.
        return "canceled-before-start"

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.send_signal(sig)
            except OSError:
                return

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        self._worker.join(timeout=15)

    # ---- device status ---------------------------------------------------------------------------------------
    def device_status(self, max_age: float = 3.0) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._device_cache
            if cached["value"] is not None:
                # A missing NPU may trigger the slower torch fallback; don't retry it
                # every poll. Successful reads refresh at ``max_age``.
                age = now - cached["at"]
                if age < (max_age if cached["value"].get("available") else 30.0):
                    return dict(cached["value"])
        value = read_device_status()
        with self._lock:
            self._device_cache = {"at": now, "value": value}
            return dict(value)


def read_device_status() -> dict[str, Any]:
    """NPU memory/utilisation from ``npu-smi info`` (cached by the manager)."""
    if os.environ.get("PIKA_WEBUI_NO_DEVICE", "").strip() == "1":
        return {"available": False, "reason": "设备探测已禁用", "devices": []}
    devices: list[dict[str, Any]] = []
    error = None
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            devices = parse_npu_smi(result.stdout)
        else:
            error = result.stderr.strip()[:200] or f"npu-smi 退出码 {result.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    if not devices:
        fallback = read_torch_device()
        if fallback:
            devices = fallback
    return {
        "available": bool(devices),
        "reason": None if devices else (error or "未检测到 NPU"),
        "devices": devices,
        "checked_at": utc_now(),
    }


def parse_npu_smi(text: str) -> list[dict[str, Any]]:
    """Parse the tabular ``npu-smi info`` output (HBM used/total per chip)."""
    devices: list[dict[str, Any]] = []
    card: int | None = None
    name = "Ascend"
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        header = re.fullmatch(r"(\d+)\s+(Ascend\S+)", parts[0]) if parts else None
        if header and len(parts) >= 2:
            card = int(header.group(1))
            name = header.group(2)
            continue
        if len(parts) < 3:
            continue
        chip_match = re.fullmatch(r"(\d+)\s+(\d+)", parts[0])
        if not chip_match or card is None:
            continue
        chip = int(chip_match.group(1))
        phy = int(chip_match.group(2))
        usage = parts[-1]
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", usage)
        hbm = None
        for used, total in pairs:
            if int(total) >= 1024:
                hbm = (int(used), int(total))
        if hbm is None:
            continue
        used, total = hbm
        devices.append({
            "id": len(devices), "card": card, "chip": chip, "phy_id": phy, "name": name,
            "label": f"NPU {card} · Chip {chip}",
            "hbm_used_mb": used, "hbm_total_mb": total,
            "hbm_used_gib": round(used / 1024, 2), "hbm_total_gib": round(total / 1024, 2),
            "hbm_percent": round(100 * used / total, 1) if total else None,
            "aicore_percent": None,
        })
    return devices


def read_torch_device() -> list[dict[str, Any]]:
    """Fallback: ask torch_npu for device memory in a short-lived child process."""
    script = (
        "import json\n"
        "out=[]\n"
        "try:\n"
        "    import torch, torch_npu\n"
        "    for i in range(torch.npu.device_count()):\n"
        "        free,total=torch.npu.mem_get_info(i)\n"
        "        out.append({'id':i,'free':int(free),'total':int(total)})\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=25)
        records = json.loads(result.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    devices = []
    for record in records:
        total = int(record.get("total") or 0)
        free = int(record.get("free") or 0)
        if total <= 0:
            continue
        used_mb = round((total - free) / 1024 / 1024)
        total_mb = round(total / 1024 / 1024)
        devices.append({
            "id": int(record.get("id", 0)), "chip": 0, "name": "Ascend",
            "hbm_used_mb": used_mb, "hbm_total_mb": total_mb,
            "hbm_used_gib": round(used_mb / 1024, 2), "hbm_total_gib": round(total_mb / 1024, 2),
            "hbm_percent": round(100 * used_mb / total_mb, 1) if total_mb else None,
            "aicore_percent": None,
        })
    return devices


# ---- HTTP helpers -------------------------------------------------------------------------------------------

def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, list[str]], dict[str, list[UploadedFile]]]:
    header = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii", "strict") + body)
    message = BytesParser(policy=email_policy).parsebytes(header)
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise SubmissionError("POST /api/tasks 需要 multipart/form-data")
    fields: dict[str, list[str]] = {}
    uploads: dict[str, list[UploadedFile]] = {}
    for index, part in enumerate(message.iter_parts()):
        if index >= 64 or part.is_multipart():
            raise SubmissionError("表单字段过多或格式无效")
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            if not filename and not payload:
                continue
            uploads.setdefault(name, []).append(
                UploadedFile(name, filename, payload, part.get_content_type()))
        else:
            charset = part.get_content_charset() or "utf-8"
            try:
                value = payload.decode(charset, errors="strict")
            except (LookupError, UnicodeError) as exc:
                raise SubmissionError("表单字段必须使用有效文本编码") from exc
            fields.setdefault(name, []).append(value)
    return fields, uploads


def _json_fields(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, (list, tuple)):
            fields[key] = [str(item) for item in value]
        elif isinstance(value, bool):
            fields[key] = ["true" if value else "false"]
        elif value is None:
            continue
        else:
            fields[key] = [str(value)]
    return fields


def _read_static(name: str) -> bytes:
    if name not in STATIC_FILES:
        raise FileNotFoundError(name)
    return (STATIC_ROOT / name).read_bytes()


class WebUIHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler_class, manager: TaskManager | None = None):
        self.task_manager = manager
        super().__init__(address, handler_class)

    def server_close(self) -> None:
        if self.task_manager is not None:
            self.task_manager.close()
        super().server_close()


class WebUIHandler(BaseHTTPRequestHandler):
    server: WebUIHTTPServer
    protocol_version = "HTTP/1.1"

    # ---- responses -------------------------------------------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str, extra: Mapping[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status >= 400:
            # An error response may leave an unread request body behind; closing the
            # connection keeps the next request on a clean parser.
            self.send_header("Connection", "close")
            self.close_connection = True
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
                   "application/json; charset=utf-8")

    def _not_found(self) -> None:
        self._json(HTTPStatus.NOT_FOUND, {"error": "未找到"})

    def _mp4(self, path: Path) -> None:
        with path.open("rb") as video:
            size = os.fstat(video.fileno()).st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                try:
                    if not match or not any(match.groups()):
                        raise ValueError
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                    else:
                        start = max(0, size - int(last))
                    if start > end or start >= size:
                        raise ValueError
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Disposition", 'inline; filename="output.mp4"')
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if getattr(self, "_head_only", False):
                return
            video.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = video.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    @staticmethod
    def _task_id(value: str) -> str | None:
        value = unquote(value)
        return value if TASK_ID_PATTERN.fullmatch(value) else None

    def _client_config(self) -> dict[str, Any]:
        release = _load_release()
        defaults = {
            "model_repo": "Pika-Labs/Pika-H3-Flex",
            "revision": release.get("model_revision", ""),
            "base_path": os.environ.get("PIKA_BASE_PATH", "models/MiniMax-H3-ref"),
            "adapter_path": os.environ.get("PIKA_ADAPTER_PATH", "models/Pika-H3-Flex"),
            "text_encoder_path": os.environ.get("PIKA_TEXT_ENCODER_PATH", ""),
        }
        return {
            "aspects": list(geometry.ASPECTS),
            "aspect_ratios": {name: list(ratios) for name, ratios in geometry.ASPECT_RATIOS.items()},
            "tiers": geometry.tier_canvases(),
            "resolutions": list(geometry.RESOLUTIONS),
            "frame_grid": {"step": 17, "offset": 5, "fps": geometry.FPS,
                           "min_frames": geometry.MIN_FRAMES, "max_frames": geometry.MAX_FRAMES,
                           "min_seconds": geometry.MIN_SECONDS, "max_seconds": geometry.MAX_SECONDS},
            "limits": {
                "images": MAX_REFERENCE_IMAGES, "videos": MAX_REFERENCE_VIDEOS,
                "audio": MAX_REFERENCE_AUDIOS, "total": MAX_REFERENCE_TOTAL,
                "image_bytes": MAX_IMAGE_BYTES, "video_bytes": MAX_VIDEO_BYTES,
                "audio_bytes": MAX_AUDIO_BYTES, "request_bytes": MAX_TOTAL_REQUEST_BYTES,
                "prompt_chars": MAX_PROMPT_LENGTH, "clip_seconds": [MIN_CLIP_SECONDS, MAX_CLIP_SECONDS],
                "clip_total_seconds": MAX_TOTAL_CLIP_SECONDS, "max_seed": MAX_SEED,
            },
            "budget": {"default": DEFAULT_BUDGET, "min": geometry.MIN_BUDGET, "max": geometry.MAX_BUDGET},
            "defaults": defaults,
            "dry_run": self.server.task_manager.dry_run,
            "max_canvas_side": geometry.MAX_CANVAS_SIDE,
            "canvas_multiple": geometry.CANVAS_MULTIPLE,
        }

    # ---- GET -------------------------------------------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        # Browsers and download managers probe with HEAD before ranged GETs.
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        manager = self.server.task_manager
        if path == "/":
            self._send(HTTPStatus.OK, _read_static("index.html"), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.OK, _read_static("favicon.svg"), "image/svg+xml")
            return
        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            try:
                body = _read_static(name)
            except (FileNotFoundError, OSError):
                self._not_found()
                return
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if content_type.startswith("text/"):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, body, content_type)
            return
        if path.startswith("/logs/"):
            task_id = self._task_id(path.removeprefix("/logs/"))
            if task_id is None or manager.get(task_id) is None:
                self._not_found()
                return
            page = _read_static("logs.html").decode("utf-8").replace("{{TASK_ID}}", task_id)
            self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/config":
            self._json(HTTPStatus.OK, self._client_config())
            return
        if path == "/api/status":
            self._json(HTTPStatus.OK, {
                "device": manager.device_status(),
                "queue": manager.queue_snapshot(),
                "dry_run": manager.dry_run,
                "active": manager.queue_snapshot().get("active"),
            })
            return
        if path == "/api/tasks":
            self._json(HTTPStatus.OK, {"tasks": manager.history()})
            return
        match = re.fullmatch(r"/api/tasks/([^/]+)(?:/(log|video|metadata))?", path)
        if not match:
            self._not_found()
            return
        task_id = self._task_id(match.group(1))
        if task_id is None or manager.get(task_id) is None:
            self._not_found()
            return
        suffix = match.group(2)
        if suffix == "log":
            query = parse_qs(parsed.query)
            try:
                tail = int(query.get("tail", ["262144"])[0])
            except ValueError:
                tail = 262144
            try:
                log = manager.read_log(task_id, tail_bytes=max(0, tail))
            except (KeyError, OSError):
                self._not_found()
                return
            self._send(HTTPStatus.OK, log.encode("utf-8"), "text/plain; charset=utf-8")
            return
        if suffix == "video":
            video = manager.artifact(task_id, "output.mp4")
            if video is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "视频尚未生成"})
                return
            self._mp4(video)
            return
        if suffix == "metadata":
            metadata = manager.artifact(task_id, "output.json")
            if metadata is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "元数据尚未生成"})
                return
            self._send(HTTPStatus.OK, metadata.read_bytes(), "application/json; charset=utf-8",
                       {"Content-Disposition": f'attachment; filename="{task_id}.json"'})
            return
        self._json(HTTPStatus.OK, manager.get(task_id))

    # ---- POST ------------------------------------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        manager = self.server.task_manager
        if path == "/api/plan":
            try:
                payload = self._read_json_body()
                plan = compute_plan_from_fields(_json_fields(payload))
            except SubmissionError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (OSError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            else:
                self._json(HTTPStatus.OK, {"plan": plan})
            return
        match = re.fullmatch(r"/api/tasks/([^/]+)/cancel", path)
        if match:
            task_id = self._task_id(match.group(1))
            if task_id is None or manager.get(task_id) is None:
                self._not_found()
                return
            try:
                result = manager.cancel(task_id)
            except TaskConflictError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except KeyError:
                self._not_found()
            else:
                self._json(HTTPStatus.ACCEPTED, {"id": task_id, "cancel": result})
            return
        if path != "/api/tasks":
            self._not_found()
            return
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "需要 Content-Length"})
            return
        if length > MAX_TOTAL_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求超过 512 MiB 限制"})
            return
        content_type = self.headers.get("Content-Type", "")
        try:
            body = self.rfile.read(length)
            if content_type.split(";", 1)[0].strip().lower() == "application/json":
                fields = _json_fields(json.loads(body.decode("utf-8")))
                uploads: dict[str, list[UploadedFile]] = {}
            else:
                fields, uploads = _parse_multipart(content_type, body)
            submission, media = validate_submission(fields, uploads)
            task_id = manager.submit(submission, media)
        except TaskConflictError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except SubmissionError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - unexpected request failure stays JSON
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._json(HTTPStatus.ACCEPTED, {"id": task_id, "seed": submission.seed,
                                             "plan": submission.plan})

    def _read_json_body(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            raise SubmissionError("需要 Content-Length")
        if length > 1024 * 1024:
            raise SubmissionError("请求过大")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise SubmissionError("JSON 请求体必须是对象")
        return payload

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # The page polls /api/status and /api/tasks every second; logging those
        # drowns the startup output and any real error. PIKA_WEBUI_ACCESS_LOG=1
        # restores the full access log.
        if os.environ.get("PIKA_WEBUI_ACCESS_LOG", "").strip() != "1":
            message = format % args
            if '"/api/status ' in message or '"/api/tasks ' in message \
                    or '"/api/tasks/' in message or '"/static/' in message:
                return
        sys.stderr.write("[webui] " + (format % args) + "\n")


def create_server(host: str = "127.0.0.1", port: int = 3000,
                  runs_dir: Path | None = None) -> WebUIHTTPServer:
    # Bind first so a second launch on an occupied port cannot mark the first
    # server's running tasks as interrupted.
    server = WebUIHTTPServer((host, port), WebUIHandler)
    try:
        server.task_manager = TaskManager(Path(runs_dir) if runs_dir else RUNS_DIR)
    except Exception:
        server.server_close()
        raise
    return server


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pika H3 Flex browser WebUI")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    server = create_server(args.host, args.port)

    def stop_on_signal(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    print(f"Pika H3 WebUI 已启动: http://{args.host}:{args.port}", flush=True)
    if server.task_manager.dry_run:
        print("PIKA_WEBUI_DRY_RUN=1：不会加载模型，只模拟进度与输出", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 WebUI", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
