# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Canvas, frame-grid and MP-budget arithmetic shared by the WebUI server and its tests.

The model imposes three hard rules, all taken from :mod:`pika_h3.core.geometry`:

* both canvas sides are multiples of ``CANVAS_MULTIPLE`` (32 px);
* the clip is ``17 n + 5`` frames at 24 fps;
* the aspect ratio is inside ``[0.25, 4.0]``.

MP mode starts from a target megapixel count, snaps the canvas onto the 32-px
lattice with the closest aspect, then picks the longest integer-second duration
whose ``width * height * frames`` product fits the configureable compute budget
(expressed in megapixel-frames, i.e. ``MP * frames``).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # the package is installed in the launch environment; keep a standalone fallback
    from pika_h3.core.geometry import (
        CANVAS_MULTIPLE as _CANVAS_MULTIPLE,
        FPS as _FPS,
        align_num_frames as _align_num_frames,
        resolve_canvas_size as _resolve_canvas_size,
    )
except Exception:  # pragma: no cover - only used when running outside the venv
    _CANVAS_MULTIPLE = 32
    _FPS = 24

    def _align_num_frames(num_frames: int) -> int:
        while num_frames % 17 != 5:
            num_frames += 1
        return num_frames

    def _resolve_canvas_size(aspect_width, aspect_height, short_edge=768,
                             max_pixels=768 * 1344, canvas_multiple=32,
                             min_aspect_ratio=0.25, max_aspect_ratio=4.0):
        ratio = aspect_width / aspect_height
        if ratio >= 1.0:
            width, height = short_edge * ratio, float(short_edge)
        else:
            width, height = float(short_edge), short_edge / ratio
        area = width * height
        if area > max_pixels:
            scale = (max_pixels / area) ** 0.5
            width, height = width * scale, height * scale
        return (max(canvas_multiple, round(height / canvas_multiple) * canvas_multiple),
                max(canvas_multiple, round(width / canvas_multiple) * canvas_multiple))


CANVAS_MULTIPLE = int(_CANVAS_MULTIPLE)
FPS = int(_FPS)
FRAME_STEP = 17
FRAME_OFFSET = 5
MIN_SECONDS = 5
MAX_SECONDS = 15
MIN_FRAMES = _align_num_frames(MIN_SECONDS * FPS)   # 124
MAX_FRAMES = _align_num_frames(MAX_SECONDS * FPS)   # 362
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0
MAX_CANVAS_SIDE = 8192
MIN_MP = 0.05
MAX_MP = 8.0
DEFAULT_BUDGET = 2000.0      # MP * frames; a soft warning threshold, not a cap
MIN_BUDGET = 20.0
MAX_BUDGET = 8000.0

ASPECTS: tuple[str, ...] = (
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1",
    "4:5", "3:4", "2:3", "9:16", "9:21",
)
ASPECT_RATIOS = {name: tuple(int(part) for part in name.split(":")) for name in ASPECTS}

BUCKETS_FILE = Path(__file__).resolve().parent.parent / "src" / "pika_h3" / "core" / "buckets.yaml"
RESOLUTIONS = ("480p", "768p")


def align_frames(num_frames: int) -> int:
    """Snap up to the next ``17 n + 5`` frame count."""
    return int(_align_num_frames(int(num_frames)))


def frames_for_duration(seconds: int | float) -> int:
    """Aligned frame count the CLI would produce for an integer-second duration."""
    return align_frames(round(float(seconds) * FPS))


def clamp_ratio(ratio: float) -> float:
    return max(MIN_ASPECT_RATIO, min(MAX_ASPECT_RATIO, float(ratio)))


def aspect_ratio(aspect: str) -> float:
    if aspect not in ASPECT_RATIOS:
        raise ValueError(f"不支持的宽高比 {aspect!r}；可选 {', '.join(ASPECTS)}")
    width, height = ASPECT_RATIOS[aspect]
    return width / height


def validate_canvas(width: int, height: int) -> None:
    """Same acceptance rule as ``conditioning.resolve_geometry``.

    ``resolve_geometry`` rejects sides that are not multiples of 32 and snaps the
    frame count to ``17n+5``; the aspect and size limits below mirror the
    pipeline's ``resolve_canvas_size`` inputs.
    """
    for name, value in (("宽度", width), ("高度", height)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name}必须是整数")
        if value <= 0:
            raise ValueError(f"{name}必须为正数")
        if value % CANVAS_MULTIPLE:
            raise ValueError(f"{name}必须是 {CANVAS_MULTIPLE} 的倍数，当前为 {value}")
        if value > MAX_CANVAS_SIDE:
            raise ValueError(f"{name}不能超过 {MAX_CANVAS_SIDE}，当前为 {value}")
    ratio = width / height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"宽高比 {ratio:.4f} 超出 [{MIN_ASPECT_RATIO:g}, {MAX_ASPECT_RATIO:g}] 范围")

def _tier_for_short_edge(short_edge: int) -> str:
    """Tier whose nominal short edge is nearest to ``short_edge``."""
    edges = tier_short_edges()
    return min(edges, key=lambda name: (abs(edges[name] - int(short_edge)), -edges[name]))


def tier_short_edges() -> dict[str, int]:
    """Nominal short edge per tier, shared by every aspect in that tier.

    The 21:9 / 9:21 rows are wider than the nominal edge, so the mode of the
    per-row short edges is the tier's identity (480 / 768 / 1088 / 1440 / 2176).
    """
    from collections import Counter

    table = _bucket_table()
    edges: dict[str, int] = {}
    for tier, rows in table["tiers"].items():
        shorts = Counter(min(int(w), int(h)) for w, h in rows)
        edges[tier] = shorts.most_common(1)[0][0]
    return edges


def nearest_bucket_canvas(mp: float, aspect: str) -> tuple[int, int, str | None, float]:
    """Closest bucket tier canvas for ``aspect`` at ``mp`` megapixels.

    Tiers are the canvas families the model was validated on and each has a
    nominal short edge shared by most of its aspect ratios (480 / 768 / 1088 /
    1440 / 2176). The requested megapixel count is matched on the short edge
    implied by the aspect ratio, not on the tier's realised area, so the tier
    choice stays consistent across neighbouring ratios: 2.0 MP 16:9 snaps to
    1920x1088, and 2.0 MP 1:1 to 1440x1440 instead of dropping to 1088x1088.

    The ultra-wide rows (21:9 and 9:21) are genuinely shorter than the nominal
    edge in every tier, so they compare against each tier's own short edge
    instead of the nominal one; otherwise 2.0 MP 21:9 would fall back a tier.

    Returns ``(width, height, tier, tier_mp)``.
    """
    table = _bucket_table()
    try:
        index = table["aspects"].index(aspect)
    except ValueError:
        width, height = snap_canvas(float(mp) * 1e6, clamp_ratio(aspect_ratio(aspect)))
        return int(width), int(height), None, width * height / 1e6
    ratio = aspect_ratio(aspect)
    target_short = math.sqrt(max(float(mp), MIN_MP) * 1e6 / ratio) if ratio >= 1.0 \
        else math.sqrt(max(float(mp), MIN_MP) * 1e6 * ratio)
    edges = tier_short_edges()
    ultra_wide = ratio >= 2.0 or ratio <= 0.5

    def distance(name: str) -> tuple[float, int]:
        if ultra_wide:
            width, height = table["tiers"][name][index]
            short = min(int(width), int(height))
        else:
            short = edges[name]
        return abs(short - target_short), -short

    tier = min(edges, key=distance)
    width, height = table["tiers"][tier][index]
    return int(width), int(height), tier, width * height / 1e6


def snap_canvas(target_area: float, ratio: float, *, aspect_tolerance: float = 0.04) -> tuple[int, int]:
    """Closest 32-px lattice canvas to ``target_area`` pixels at ``ratio``.

    ``resolve_canvas_size`` produces the starting short edge; the candidate then
    moves in whole 32-px steps to minimise the area error while keeping the
    aspect error inside ``aspect_tolerance`` (4 %). If no lattice point meets the
    tolerance the closest combined score is used.
    """
    m = CANVAS_MULTIPLE
    ratio = clamp_ratio(ratio)
    target_area = max(float(m * m), float(target_area))
    short_edge = int(round(math.sqrt(target_area / max(ratio, 1.0 / ratio))))
    base_h, base_w = _resolve_canvas_size(
        ratio, 1.0, short_edge=max(m, short_edge), max_pixels=max(m * m, int(round(target_area))),
        canvas_multiple=m)
    base_w = max(m, int(base_w))
    base_h = max(m, int(base_h))
    base_w = int(round(base_w / m)) * m
    base_h = int(round(base_h / m)) * m

    candidates: list[tuple[float, float, int, int]] = []
    fallback: list[tuple[float, float, int, int]] = []
    for dw in range(-8, 9):
        for dh in range(-8, 9):
            width = (base_w // m + dw) * m
            height = (base_h // m + dh) * m
            if width < m or height < m or width > MAX_CANVAS_SIDE or height > MAX_CANVAS_SIDE:
                continue
            if not MIN_ASPECT_RATIO <= width / height <= MAX_ASPECT_RATIO:
                continue
            aspect_error = abs(math.log((width / height) / ratio))
            area_error = abs(math.log((width * height) / target_area))
            entry = (area_error + 0.5 * aspect_error, area_error, width, height)
            fallback.append(entry)
            if aspect_error <= aspect_tolerance:
                candidates.append(entry)
    pool = candidates or fallback
    if not pool:
        raise ValueError(f"在宽高比 {ratio:.4f} 附近找不到 {target_area:.0f} 像素的 32 px 画布")
    pool.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, width, height = pool[0]
    return int(width), int(height)


def plan_canvas(
    *,
    mp: float,
    aspect: str,
    budget: float = DEFAULT_BUDGET,
    canvas: tuple[int, int] | None = None,
    duration: int | None = None,
    allow_shrink: bool = False,
) -> dict[str, Any]:
    """Resolve canvas + aligned frame count + duration for an MP-mode request.

    With no ``duration`` the largest integer-second duration from 5 s to 15 s
    whose ``width*height*frames`` product fits ``budget`` (MP-frames) is chosen.
    If even the 5 s clip does not fit, this raises unless ``allow_shrink`` is set,
    in which case the canvas is scaled down. Silent canvas substitution is off by
    default so a requested MP/aspect is never quietly replaced by a smaller
    canvas. An explicit ``duration`` is
    honoured and only produces a warning when it exceeds the budget.
    """
    ratio = aspect_ratio(aspect)
    notes: list[str] = []
    reduced = False
    manual_canvas = canvas is not None
    tier: str | None = None

    if manual_canvas:
        width, height = int(canvas[0]), int(canvas[1])
        validate_canvas(width, height)
        canvas_source = "manual"
        mp_value = width * height / 1e6
        if abs(width / height - ratio) > 0.02:
            notes.append(
                f"手动画布 {width}x{height} 的宽高比 {width / height:.3f} 偏离所选 {aspect} "
                f"({ratio:.3f})")
        if mp > 0 and abs(mp_value - mp) > max(0.02, 0.1 * mp):
            notes.append(f"手动画布为 {mp_value:.3f} MP，与输入的 {mp:g} MP 不同")
    else:
        mp_value = max(MIN_MP, min(MAX_MP, float(mp)))
        if abs(mp_value - float(mp)) > 1e-9:
            notes.append(f"MP 已限制在 [{MIN_MP:g}, {MAX_MP:g}] 范围内")
        # Snap to the nearest bucket tier for this aspect, matching the canvas
        # families ComfyUI exposes. Free-form search would land on odd sizes like
        # 1888x1056 for a 2.0 MP request instead of the 1080p canvas.
        width, height, tier, tier_mp = nearest_bucket_canvas(mp_value, aspect)
        canvas_source = "mp"
        if tier is not None:
            notes.append(
                f"已对齐到 {tier} 档位 {width}x{height}（{tier_mp:.3f} MP，"
                f"与输入 {mp_value:g} MP 相差 {(tier_mp - mp_value) / mp_value * 100:+.1f}%）")

    area = width * height
    options = [(seconds, frames_for_duration(seconds))
               for seconds in range(MIN_SECONDS, MAX_SECONDS + 1)]

    if duration is not None:
        seconds = int(duration)
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"时长必须在 {MIN_SECONDS}–{MAX_SECONDS} 秒之间")
        frames = frames_for_duration(seconds)
        budget_used = area * frames / 1e6
        if budget_used > budget + 1e-6:
            notes.append(
                f"手动时长超出预算：{width}x{height} x {frames} 帧 = {budget_used:.1f} MP·帧 "
                f"> {budget:.1f} MP·帧，可能变慢或显存不足")
    else:
        fitting = [(sec, frames) for sec, frames in options if area * frames <= budget * 1e6 + 1e-6]
        if fitting:
            seconds, frames = fitting[-1]
        else:
            if manual_canvas:
                seconds, frames = MIN_SECONDS, frames_for_duration(MIN_SECONDS)
                notes.append(
                    f"画布超出预算：{width}x{height} x {frames} 帧 = "
                    f"{area * frames / 1e6:.1f} MP·帧 > {budget:.1f} MP·帧，已保留手动画布")
            elif not allow_shrink:
                # Silent substitution would hand back a different canvas than the
                # user asked for, so refuse and let them lower MP or raise budget.
                frames = frames_for_duration(MIN_SECONDS)
                needed = area * frames / 1e6
                raise ValueError(
                    f"{width}x{height} 连 {MIN_SECONDS}s（{frames} 帧）都需要 "
                    f"{needed:.1f} MP·帧，超出预算 {budget:.1f} MP·帧；"
                    f"请降低 MP、提高预算，或显式开启画布回退")
            else:
                # Shrink the canvas so the shortest allowed clip fits the budget.
                frames = frames_for_duration(MIN_SECONDS)
                allowed_area = budget * 1e6 / frames
                width, height = snap_canvas(allowed_area, ratio)
                area = width * height
                seconds = MIN_SECONDS
                reduced = True
                mp_value = area / 1e6
                notes.append(
                    f"目标 {mp:g} MP 在预算内无法生成长度 ≥{MIN_SECONDS}s 的视频，"
                    f"画布已自动缩小到 {width}x{height}（{mp_value:.3f} MP）")
        budget_used = area * frames / 1e6

    if mp_value > MAX_MP:
        notes.append("画布超过建议的 8 MP，生成时间与显存占用会显著增加")

    return {
        "canvas_mode": "mp" if canvas_source in {"mp", "manual"} else "preset",
        "canvas_source": canvas_source,
        "bucket_tier": tier if canvas_source == "mp" else _tier_for_short_edge(min(width, height)),
        "aspect": aspect,
        "aspect_ratio": round(width / height, 6),
        "aspect_target": round(ratio, 6),
        "width": width,
        "height": height,
        "mp_requested": round(float(mp), 4),
        "mp": round(area / 1e6, 4),
        "frames": int(frames),
        "duration_seconds": int(seconds),
        "seconds": round(frames / FPS, 4),
        "fps": FPS,
        "budget": round(float(budget), 2),
        "budget_used": round(budget_used, 2),
        "budget_fraction": round(budget_used / budget, 4) if budget else None,
        "auto_duration": duration is None,
        "reduced": reduced,
        "notes": notes,
        "frame_grid": "17n+5",
    }


@lru_cache(maxsize=1)
def _bucket_table() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(BUCKETS_FILE.read_text(encoding="utf-8"))


def bucket_canvas(resolution: str, aspect: str) -> tuple[int, int]:
    """Preset-path canvas straight from the packaged ``buckets.yaml``."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"分辨率必须为 {', '.join(RESOLUTIONS)}")
    table = _bucket_table()
    tier = "p" + resolution[:-1]
    try:
        index = table["aspects"].index(aspect)
    except ValueError:
        raise ValueError(f"不支持的宽高比 {aspect!r}") from None
    width, height = table["tiers"][tier][index]
    return int(width), int(height)


def preset_plan(resolution: str, aspect: str, duration: int, *, budget: float = DEFAULT_BUDGET) -> dict[str, Any]:
    """Preset path: bucket canvas + explicit (validated) duration."""
    if not MIN_SECONDS <= int(duration) <= MAX_SECONDS:
        raise ValueError(f"时长必须在 {MIN_SECONDS}–{MAX_SECONDS} 秒之间")
    width, height = bucket_canvas(resolution, aspect)
    frames = frames_for_duration(int(duration))
    used = width * height * frames / 1e6
    notes = []
    if used > budget + 1e-6:
        notes.append(
            f"该预设 {width}x{height} x {frames} 帧 = {used:.1f} MP·帧 超出预算 "
            f"{budget:.1f} MP·帧")
    return {
        "canvas_mode": "preset",
        "canvas_source": "bucket",
        "resolution": resolution,
        "aspect": aspect,
        "aspect_ratio": round(width / height, 6),
        "aspect_target": round(aspect_ratio(aspect), 6),
        "width": width,
        "height": height,
        "mp_requested": None,
        "mp": round(width * height / 1e6, 4),
        "frames": int(frames),
        "duration_seconds": int(duration),
        "seconds": round(frames / FPS, 4),
        "fps": FPS,
        "budget": round(float(budget), 2),
        "budget_used": round(used, 2),
        "budget_fraction": round(used / budget, 4) if budget else None,
        "auto_duration": False,
        "reduced": False,
        "notes": notes,
        "frame_grid": "17n+5",
    }


def tier_canvases() -> dict[str, dict[str, list[int]]]:
    """``{resolution: {aspect: [width, height]}}`` for the UI preset picker."""
    table = _bucket_table()
    out: dict[str, dict[str, list[int]]] = {}
    for resolution in RESOLUTIONS:
        tier = "p" + resolution[:-1]
        out[resolution] = {
            aspect: [int(width), int(height)]
            for aspect, (width, height) in zip(table["aspects"], table["tiers"][tier])
        }
    return out
