# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""WebUI checks: MP arithmetic, reference limits, --canvas plumbing, seed rules.

Run with::

    .venv312/bin/python -m unittest webui.tests.test_webui -v
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from webui import geometry, server


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class MPCanvasTests(unittest.TestCase):
    """Requirement 2 of the test list: MP -> canvas table."""

    CASES = (0.4, 0.8, 1.0, 1.5)

    def test_mp_mode_duration_is_user_choice(self) -> None:
        # Duration is explicit in MP mode; the budget only warns.
        fields = {"canvas_mode": ["mp"], "mp": ["2.0"], "aspect": ["16:9"],
                  "duration": ["10"], "budget": ["260"]}
        plan = server.compute_plan_from_fields(fields)
        self.assertEqual(plan["duration_seconds"], 10)
        self.assertEqual(plan["frames"], 243)
        self.assertFalse(plan["auto_duration"])
        self.assertTrue(any("超出预算" in note for note in plan["notes"]))
        # A different user-chosen duration changes the frame count, not the canvas.
        fields["duration"] = ["5"]
        shorter = server.compute_plan_from_fields(fields)
        self.assertEqual((shorter["width"], shorter["height"]), (plan["width"], plan["height"]))
        self.assertEqual(shorter["frames"], 124)

    def test_mp_mode_requires_duration(self) -> None:
        with self.assertRaises(server.SubmissionError):
            server.compute_plan_from_fields(
                {"canvas_mode": ["mp"], "mp": ["1.0"], "aspect": ["16:9"]})


        # MP requests snap to a bucket tier, so the realised area can differ from
        # the request by a whole tier step (for example 0.4 MP 3:2 -> 704x480,
        # -15.5%). Assert lattice/ratio correctness and tier membership instead
        # of a tight area tolerance.
        tiers = geometry.tier_short_edges()
        rows = []
        for mp in self.CASES:
            for aspect in geometry.ASPECTS:
                plan = geometry.plan_canvas(mp=mp, aspect=aspect, budget=geometry.MAX_BUDGET,
                                            duration=geometry.MIN_SECONDS)
                width, height = plan["width"], plan["height"]
                self.assertEqual(width % 32, 0, f"{mp} MP {aspect}: width {width}")
                self.assertEqual(height % 32, 0, f"{mp} MP {aspect}: height {height}")
                # The canvas must be exactly one of the bucket rows for that tier.
                table = geometry._bucket_table()
                row = table["tiers"][plan["bucket_tier"]][table["aspects"].index(aspect)]
                self.assertEqual((width, height), (int(row[0]), int(row[1])),
                                 f"{mp} MP {aspect}: not a {plan['bucket_tier']} row")
                ratio = width / height
                target_ratio = geometry.aspect_ratio(aspect)
                aspect_error = abs(ratio / target_ratio - 1)
                # Tier rows are themselves approximations of the nominal ratio.
                self.assertLessEqual(aspect_error, 0.10, f"{mp} MP {aspect}: aspect {ratio}")
                self.assertEqual(plan["frames"] % 17, 5)
                rows.append((mp, aspect, f"{width}x{height}", plan["frames"], aspect_error * 100))
        self.assertGreater(len(rows), 0)

    def test_mp_matches_short_edge_tier(self) -> None:
        # The tier is chosen on the short edge implied by MP and aspect.
        self.assertEqual(geometry.nearest_bucket_canvas(2.0, "16:9")[:3], (1920, 1088, "p1080"))
        self.assertEqual(geometry.nearest_bucket_canvas(0.4, "16:9")[:3], (832, 480, "p480"))
        self.assertEqual(geometry.nearest_bucket_canvas(1.0, "16:9")[:3], (1344, 768, "p768"))
        self.assertEqual(geometry.nearest_bucket_canvas(2.0, "1:1")[:3], (1440, 1440, "p1440"))
        self.assertEqual(geometry.nearest_bucket_canvas(2.0, "21:9")[:3], (2208, 960, "p1080"))
        self.assertEqual(geometry.nearest_bucket_canvas(3.7, "16:9")[:3], (2560, 1440, "p1440"))
        self.assertEqual(geometry.nearest_bucket_canvas(8.0, "16:9")[:3], (3840, 2176, "p2160"))

    def test_plan_accepts_resolve_geometry(self) -> None:
        from pika_h3.core.conditioning import Request, resolve_geometry

        for mp in self.CASES:
            for aspect in geometry.ASPECTS:
                plan = geometry.plan_canvas(mp=mp, aspect=aspect, budget=geometry.MAX_BUDGET,
                                            duration=10)
                request = Request(task="t2va", prompt="x", height=plan["height"],
                                  width=plan["width"], num_frames=plan["frames"])
                geom = resolve_geometry(request)
                self.assertEqual((geom.height, geom.width), (plan["height"], plan["width"]))
                self.assertEqual(geom.num_frames, plan["frames"])
                self.assertEqual(geom.num_frames % 17, 5)

    def test_budget_picks_largest_fitting_duration(self) -> None:
        plan = geometry.plan_canvas(mp=1.0, aspect="16:9", budget=260)
        area = plan["width"] * plan["height"]
        self.assertLessEqual(area * plan["frames"], 260 * 1e6)
        longer = geometry.frames_for_duration(plan["duration_seconds"] + 1)
        if plan["duration_seconds"] < geometry.MAX_SECONDS:
            self.assertGreater(area * longer, 260 * 1e6)

    def test_budget_too_small_rejects_by_default(self) -> None:
        # Shrinking is opt-in so a requested canvas is never silently replaced.
        with self.assertRaises(ValueError):
            geometry.plan_canvas(mp=1.5, aspect="16:9", budget=60)

    def test_budget_too_small_reduces_canvas_when_allowed(self) -> None:
        plan = geometry.plan_canvas(mp=1.5, aspect="16:9", budget=60, allow_shrink=True)
        self.assertTrue(plan["reduced"])
        self.assertLessEqual(plan["width"] * plan["height"] * plan["frames"], 60 * 1e6 + 1e6)

    def test_override_is_preserved(self) -> None:
        plan = geometry.plan_canvas(mp=1.0, aspect="16:9", budget=260,
                                    canvas=(1024, 576), duration=5)
        self.assertEqual((plan["width"], plan["height"]), (1024, 576))
        self.assertEqual(plan["duration_seconds"], 5)

    def test_invalid_manual_canvas_rejected(self) -> None:
        with self.assertRaises(ValueError):
            geometry.plan_canvas(mp=1.0, aspect="16:9", canvas=(1000, 576))
        with self.assertRaises(ValueError):
            geometry.plan_canvas(mp=1.0, aspect="16:9", canvas=(64, 1024))

    def test_preset_uses_buckets(self) -> None:
        plan = geometry.preset_plan("480p", "21:9", 5)
        self.assertEqual((plan["width"], plan["height"]), (960, 416))
        self.assertEqual(plan["frames"], 124)
        self.assertEqual(plan["canvas_mode"], "preset")

    def test_metadata_records_canvas_source(self) -> None:
        """``pipeline.generate`` metadata must advertise the explicit canvas as such."""
        import pika_h3.pipeline as pipeline
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        self.assertIn('"source": canvas["canvas_source"]', source)
        self.assertIn('"bucket_tier": canvas["tier"]', source)

    def test_frame_grid_boundaries(self) -> None:
        self.assertEqual(geometry.frames_for_duration(5), 124)
        self.assertEqual(geometry.frames_for_duration(10), 243)
        self.assertEqual(geometry.frames_for_duration(15), 362)
        for frames in (geometry.MIN_FRAMES, geometry.MAX_FRAMES):
            self.assertEqual(frames % 17, 5)

    def test_all_aspects_in_buckets(self) -> None:
        import yaml
        table = yaml.safe_load(geometry.BUCKETS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(tuple(table["aspects"]), geometry.ASPECTS)
        # buckets.yaml lists the raw 17n+5 grid; the CLI snaps an integer-second
        # request up to the next 17n+5, so ours must match that (not the raw list).
        for frames in (int(f) for f in table["frames"]):
            self.assertEqual(frames % 17, 5)
        from pika_h3.core import geometry as core_geometry
        for seconds in range(5, 16):
            self.assertEqual(geometry.frames_for_duration(seconds),
                             core_geometry.align_num_frames(seconds * 24))


def _upload(field: str, name: str, payload: bytes) -> server.UploadedFile:
    return server.UploadedFile(field, name, payload)


PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000154a24f0d0000000049454e44ae426082")
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
WAV_HEADER = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt "


def _make_clip(path: Path, seconds: float, *, kind: str = "video") -> Path:
    """Real 1-second/2-second media via ffmpeg for the duration rule tests."""
    if kind == "video":
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            f"color=c=black:s=64x64:d={seconds}:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]
    else:
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a", "aac", str(path),
        ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    return path


class ReferenceLimitTests(unittest.TestCase):
    """Requirement 3 of the test list: ref2va limits."""

    def fields(self, **overrides: str) -> dict[str, list[str]]:
        base = {
            "mode": ["references"],
            "prompt": ["test reference prompt"],
            "canvas_mode": ["mp"],
            "mp": ["0.4"],
            "aspect": ["16:9"],
            "duration": ["5"],
            "budget": ["260"],
            "seed": ["7"],
            "low_vram": ["true"],
        }
        base.update({key: [value] for key, value in overrides.items()})
        return base

    def assert_rejected(self, uploads, fragment: str) -> None:
        with self.assertRaises(server.SubmissionError) as caught:
            server.validate_submission(self.fields(), uploads)
        self.assertIn(fragment, str(caught.exception))

    def test_ten_images_rejected(self) -> None:
        uploads = {"refs": [_upload("refs", f"{index}.png", PNG_1PX) for index in range(10)]}
        self.assert_rejected(uploads, "最多 9 张参考图片")

    def test_four_videos_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clip = _make_clip(tmp_path / "clip.mp4", 2.0)
            payload = clip.read_bytes()
            uploads = {"video_refs": [_upload("video_refs", f"v{index}.mp4", payload)
                                      for index in range(4)]}
            self.assert_rejected(uploads, "最多 3 个参考视频")

    def test_thirteen_total_rejected(self) -> None:
        images = [_upload("refs", f"{index}.png", PNG_1PX) for index in range(9)]
        videos = [_upload("video_refs", f"v{index}.mp4", MP4_BYTES) for index in range(3)]
        audios = [_upload("audio_refs", f"a{index}.wav", WAV_HEADER) for index in range(1)]
        self.assert_rejected({"refs": images, "video_refs": videos, "audio_refs": audios},
                             "最多 12 个")

    def test_one_second_video_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clip = _make_clip(tmp_path / "short.mp4", 1.0)
            uploads = {
                "refs": [_upload("refs", "x.png", PNG_1PX)],
                "video_refs": [_upload("video_refs", "short.mp4", clip.read_bytes())],
            }
            self.assert_rejected(uploads, "必须为 2–15 秒")

    def test_two_second_video_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clip = _make_clip(tmp_path / "ok.mp4", 2.0)
            uploads = {
                "refs": [_upload("refs", "x.png", PNG_1PX)],
                "video_refs": [_upload("video_refs", "ok.mp4", clip.read_bytes())],
            }
            submission, media = server.validate_submission(self.fields(), uploads)
            self.assertEqual(media["counts"]["video"], 1)
            self.assertTrue(2.0 <= media["durations"]["video"][0] <= 2.2)

    def test_audio_only_rejected(self) -> None:
        uploads = {"audio_refs": [_upload("audio_refs", "a.wav", WAV_HEADER)]}
        self.assert_rejected(uploads, "至少需要一张参考图片或一个参考视频")

    def test_combined_video_duration_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clip = _make_clip(tmp_path / "long.mp4", 8.0)
            payload = clip.read_bytes()
            uploads = {
                "refs": [_upload("refs", "x.png", PNG_1PX)],
                "video_refs": [_upload("video_refs", "a.mp4", payload),
                               _upload("video_refs", "b.mp4", payload)],
            }
            self.assert_rejected(uploads, "参考视频总时长")

    def test_sixteen_seconds_single_clip_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clip = _make_clip(tmp_path / "long.mp4", 16.0)
            uploads = {
                "refs": [_upload("refs", "x.png", PNG_1PX)],
                "video_refs": [_upload("video_refs", "long.mp4", clip.read_bytes())],
            }
            self.assert_rejected(uploads, "必须为 2–15 秒")


class KeyframeModeTests(unittest.TestCase):
    def test_keyframe_mode_maps_task(self) -> None:
        base = {"mode": ["keyframes"], "prompt": ["x"], "canvas_mode": ["mp"], "mp": ["0.4"],
                "aspect": ["16:9"], "duration": ["5"], "budget": ["260"], "seed": ["1"], "low_vram": ["true"]}
        first = {"first": [_upload("first", "a.png", PNG_1PX)]}
        last = {"last": [_upload("last", "b.png", PNG_1PX)]}
        self.assertEqual(server.validate_submission(base, first)[0].task, "i2va")
        self.assertEqual(server.validate_submission(base, last)[0].task, "l2va")
        self.assertEqual(server.validate_submission(base, {**first, **last})[0].task, "fl2va")

    def test_firstlast_requires_both(self) -> None:
        fields = {"mode": ["firstlast"], "prompt": ["x"], "canvas_mode": ["mp"], "mp": ["0.4"],
                  "aspect": ["16:9"], "duration": ["5"], "budget": ["260"], "seed": ["1"], "low_vram": ["true"]}
        with self.assertRaises(server.SubmissionError):
            server.validate_submission(fields, {"first": [_upload("first", "a.png", PNG_1PX)]})


class SeedTests(unittest.TestCase):
    """Seed control: server draws/echoes, fixed seeds are reused."""

    def fields(self, seed: str | None = None) -> dict[str, list[str]]:
        data = {
            "mode": ["text"], "prompt": ["x"], "canvas_mode": ["mp"], "mp": ["0.4"],
            "aspect": ["16:9"], "duration": ["5"], "budget": ["260"], "low_vram": ["true"],
        }
        if seed is not None:
            data["seed"] = [seed]
        return data

    def test_missing_seed_is_drawn(self) -> None:
        submission, _ = server.validate_submission(self.fields(), {})
        self.assertGreaterEqual(submission.seed, 0)
        self.assertLess(submission.seed, 2**63)
        self.assertTrue(submission.seed != 0 or server.random_seed() != 0)

    def test_invalid_seed_rejected(self) -> None:
        for bad in ("-1", "2^63", "9223372036854775808", "abc", "1.5"):
            with self.assertRaises(server.SubmissionError, msg=bad):
                server.validate_submission(self.fields(bad), {})

    def test_max_seed_accepted(self) -> None:
        submission, _ = server.validate_submission(self.fields(str(2**63 - 1)), {})
        self.assertEqual(submission.seed, 2**63 - 1)

    def test_explicit_seed_echoed(self) -> None:
        submission, _ = server.validate_submission(self.fields("12345"), {})
        self.assertEqual(submission.seed, 12345)
        task = server.Task(id="20260101-000000-deadbeef", mode="text", task="t2va",
                           prompt="x", plan=submission.plan, seed=submission.seed,
                           low_vram=True, tensor_parallel=False, model={})
        self.assertEqual(task.to_public()["seed"], 12345)


class CanvasCliTests(unittest.TestCase):
    """Requirement 4 of the test list: --canvas plumbing without the model."""

    def test_canvas_parser(self) -> None:
        from pika_h3.cli import canvas_size

        self.assertEqual(canvas_size("1024x576"), (1024, 576))
        for bad in ("1000x576", "1024", "32x2000", "0x0"):
            with self.assertRaises(Exception):
                canvas_size(bad)

    def test_resolve_canvas_args_explicit(self) -> None:
        from pika_h3.cli import parser
        from pika_h3.pipeline import resolve_canvas_args

        args = parser().parse_args([
            "--task", "t2va", "--prompt", "x", "--canvas", "1024x576", "--duration", "5",
            "--output", "/tmp/x.mp4", "--revision", "0" * 40,
        ])
        resolved = resolve_canvas_args(args)
        self.assertEqual((resolved["width"], resolved["height"]), (1024, 576))
        self.assertEqual(resolved["num_frames"], 124)
        self.assertIsNone(resolved["tier"])
        self.assertEqual(resolved["canvas_source"], "explicit")

    def test_resolve_canvas_args_bucket_unchanged(self) -> None:
        from pika_h3.cli import parser
        from pika_h3.pipeline import resolve_canvas_args

        args = parser().parse_args([
            "--task", "t2va", "--prompt", "x", "--resolution", "480p", "--aspect", "21:9",
            "--duration", "15", "--output", "/tmp/x.mp4", "--revision", "0" * 40,
        ])
        resolved = resolve_canvas_args(args)
        self.assertEqual((resolved["width"], resolved["height"]), (960, 416))
        self.assertEqual(resolved["tier"], "p480")
        self.assertEqual(resolved["num_frames"], 362)

    def test_build_command_uses_canvas(self) -> None:
        task = server.Task(
            id="20260101-000000-deadbeef", mode="text", task="t2va", prompt="x",
            plan={"width": 1024, "height": 576, "frames": 124, "duration_seconds": 5},
            seed=42, low_vram=True, tensor_parallel=False,
            model={"repo": "r", "revision": "0" * 40, "base_path": "", "adapter_path": "",
                   "text_encoder_path": ""},
        )
        with tempfile.TemporaryDirectory() as tmp:
            command = server.build_command(task, Path(tmp))
        self.assertIn("--canvas", command)
        self.assertEqual(command[command.index("--canvas") + 1], "1024x576")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertIn("--low-vram", command)

    def test_pipeline_generate_uses_resolved_canvas(self) -> None:
        """``generate`` must take width/height/frames from ``resolve_canvas_args``.

        The pure helper is exercised above; this checks the wiring in the pipeline
        without importing the model stack (the heavy imports are inside ``generate``).
        """
        import pika_h3.pipeline as pipeline
        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        self.assertIn("canvas = resolve_canvas_args(args)", source)
        self.assertIn("width, height, frames = canvas[", source)
        self.assertIn("build_request(", source)
        # The bucket lookup lives only inside resolve_canvas_args' non-explicit branch.
        self.assertEqual(source.count('table["tiers"][tier]'), 1)

    def test_build_request_uses_canvas(self) -> None:
        """``pipeline.build_request`` must put the explicit canvas on ``Request``."""
        import types
        import pika_h3.pipeline as pipeline
        from pika_h3.cli import parser

        args = parser().parse_args([
            "--task", "t2va", "--prompt", "x", "--canvas", "1024x576", "--duration", "5",
            "--output", "/tmp/x.mp4", "--revision", "0" * 40,
        ])
        canvas = pipeline.resolve_canvas_args(args)
        request = pipeline.build_request(args, canvas, [])
        self.assertEqual((request.width, request.height), (1024, 576))
        self.assertEqual(request.num_frames, 124)
        self.assertEqual(request.task, "t2va")
        from pika_h3.core.conditioning import resolve_geometry
        geom = resolve_geometry(request)
        self.assertEqual((geom.width, geom.height), (1024, 576))


class StaticChecks(unittest.TestCase):
    def test_no_eval_or_exec_in_js(self) -> None:
        for path in sorted((REPO_ROOT / "webui" / "static").glob("*.js")):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\beval\s*\(", path.name)
            self.assertNotRegex(text, r"\bexec\s*\(", path.name)
            self.assertNotIn("new Function(", text, path.name)


class ProgressMarkerTests(unittest.TestCase):
    """The child's ``[progress]`` markers drive the task state."""

    def test_marker_parsing(self) -> None:
        manager = server.TaskManager.__new__(server.TaskManager)
        manager.runs_dir = Path(tempfile.mkdtemp())
        manager._lock = threading.RLock()
        manager._condition = threading.Condition(manager._lock)
        manager._tasks = {}
        manager._pending = __import__("collections").deque()
        manager._active_id = None
        manager._closed = False
        manager._process = None
        manager.dry_run = True
        manager.dry_run_delay = 0.0
        manager._device_cache = {"at": 0.0, "value": None}
        task = server.Task(id="20260101-000000-deadbeef", mode="text", task="t2va",
                           prompt="x", plan={"width": 832, "height": 480, "frames": 124,
                                             "duration_seconds": 5},
                           seed=1, low_vram=True, tensor_parallel=False, model={},
                           status="running", created_at=server.utc_now())
        manager._tasks[task.id] = task
        run = manager.runs_dir / task.id
        run.mkdir(parents=True)
        (run / "task.log").write_text("")
        manager._record_line(task.id, "[progress] stage=conditioning pct=40 what=done")
        self.assertEqual((task.stage, task.progress), ("conditioning", 40))
        manager._record_line(task.id, "[progress] stage=sampling pct=71 step=3 total=6")
        self.assertEqual((task.stage, task.progress), ("sampling", 71))
        # A late ``done`` marker must not mark the task finished while the child runs.
        manager._record_line(task.id, "[progress] stage=done pct=100 output=/x.mp4")
        self.assertEqual((task.stage, task.status), ("encode", "running"))
        self.assertLess(task.progress, 100)
        # Progress is monotonic.
        manager._record_line(task.id, "[progress] stage=prepare pct=2")
        self.assertEqual(task.progress, 99)
        # Human-readable fallbacks still move the stage.
        manager._record_line(task.id, "[cond] Qwen3-VL loaded in 1.0s (offload=True)")
        self.assertEqual(task.stage, "loading")
        manager._record_line(task.id, "[cond] VAEs loaded in 1.0s (offload=True)")
        self.assertEqual(task.stage, "conditioning")

    def test_seed_echoed_in_task_record(self) -> None:
        manager = server.TaskManager.__new__(server.TaskManager)
        manager.runs_dir = Path(tempfile.mkdtemp())
        manager._lock = threading.RLock()
        manager._condition = threading.Condition(manager._lock)
        manager._tasks = {}
        manager._pending = __import__("collections").deque()
        manager._active_id = None
        manager._closed = False
        manager._process = None
        manager.dry_run = True
        manager.dry_run_delay = 0.0
        manager._device_cache = {"at": 0.0, "value": None}
        task = server.Task(id="20260101-000000-deadbeef", mode="text", task="t2va",
                           prompt="x", plan={"width": 832, "height": 480, "frames": 124,
                                             "duration_seconds": 5},
                           seed=424242, low_vram=True, tensor_parallel=False, model={},
                           created_at=server.utc_now())
        manager._tasks[task.id] = task
        public = manager.get(task.id)
        self.assertEqual(public["seed"], 424242)


if __name__ == "__main__":
    unittest.main()
