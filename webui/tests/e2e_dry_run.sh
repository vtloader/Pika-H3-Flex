#!/usr/bin/env bash
# End-to-end dry-run checks for the WebUI. Usage: bash webui/tests/e2e_dry_run.sh [port]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv312/bin/python"
PORT="${1:-3111}"
BASE="http://127.0.0.1:$PORT"
RUNS="${PIKA_WEBUI_RUNS_DIR:-/tmp/pika-webui-e2e}"

start_server() {
  rm -rf "$RUNS"
  PIKA_WEBUI_DRY_RUN=1 PIKA_WEBUI_RUNS_DIR="$RUNS" HOST=127.0.0.1 PORT="$PORT" \
    setsid nohup "$PY" -m webui.server > /tmp/pika-webui-e2e.log 2>&1 < /dev/null &
  SERVER_PID=$!
  for _ in $(seq 1 50); do
    if curl -sf -o /dev/null "$BASE/"; then return 0; fi
    sleep 0.2
  done
  echo "server did not start"; cat /tmp/pika-webui-e2e.log; exit 1
}
stop_server() {
  kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
  sleep 0.5
}
trap stop_server EXIT

start_server

echo "### 1. main page"
curl -sS -o /tmp/e2e-index.html -w "GET / -> %{http_code}, %{size_download} bytes, %{content_type}\n" "$BASE/"
grep -o '<title>[^<]*</title>' /tmp/e2e-index.html

echo
echo "### 2. static assets"
for asset in /static/app.js /static/logs.js /static/seed.js /static/style.css; do
  curl -sS -o /dev/null -w "$asset -> %{http_code} (%{size_download} bytes)\n" "$BASE$asset"
done

echo
echo "### 3. config + plan"
curl -sS "$BASE/api/config" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("aspects:", len(d["aspects"]), "tiers:", list(d["tiers"]), "limits:", d["limits"]["images"], d["limits"]["videos"], d["limits"]["audio"], d["limits"]["total"])'
curl -sS -X POST "$BASE/api/plan" -H 'Content-Type: application/json' \
  -d '{"canvas_mode":"mp","mp":"1.0","aspect":"16:9","duration":"5","budget":"260"}' \
  | "$PY" -c 'import json,sys; p=json.load(sys.stdin)["plan"]; print("plan:", "%dx%d" % (p["width"], p["height"]), p["frames"], "frames", p["seconds"], "s", "budget", p["budget_used"])'

echo
echo "### 4. submit task 1 (no seed: server draws one)"
TASK1=$(curl -sS -X POST "$BASE/api/tasks" \
  -F 'mode=text' -F 'prompt=一只红色的球在木桌上弹跳，暖色午后光线。' \
  -F 'canvas_mode=mp' -F 'mp=1.0' -F 'aspect=16:9' -F 'duration=5' -F 'budget=260' \
  -F 'low_vram=true' -F 'tensor_parallel=false')
echo "$TASK1" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("202 id:", d["id"], "seed:", d["seed"], "canvas:", "%dx%d" % (d["plan"]["width"], d["plan"]["height"]), "frames:", d["plan"]["frames"])'
ID1=$(echo "$TASK1" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
SEED1=$(echo "$TASK1" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["seed"])')

echo
echo "### 5. poll progress + log"
for _ in $(seq 1 30); do
  STATE=$(curl -sS "$BASE/api/tasks/$ID1")
  STATUS=$(echo "$STATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  if [ "$STATUS" = "done" ] || [ "$STATUS" = "error" ]; then break; fi
  echo "$STATE" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("  status:", d["status"], "stage:", d["stage"], "progress:", d["progress"], "seed:", d["seed"])'
  sleep 1
done
echo "$STATE" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("final:", d["status"], d["stage"], d["progress"], "elapsed:", d["elapsed_seconds"], "s", "seed:", d["seed"], "output:", d["output_ready"], "json:", d["metadata_ready"])'

echo
echo "--- log (with [progress] markers) ---"
curl -sS "$BASE/api/tasks/$ID1/log" | sed -n '1,14p'
echo "..."
curl -sS "$BASE/api/tasks/$ID1/log" | grep -c '^\[progress\]' | xargs echo "progress marker lines:"

echo
echo "### 6. output MP4 + metadata"
curl -sS -o /tmp/e2e.mp4 -w "GET video -> %{http_code}, %{size_download} bytes, %{content_type}\n" "$BASE/api/tasks/$ID1/video"
ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 /tmp/e2e.mp4
curl -sS -D /tmp/e2e-meta-headers.txt -o /tmp/e2e.json -w "GET metadata -> %{http_code}, %{size_download} bytes\n" "$BASE/api/tasks/$ID1/metadata"
grep -i 'content-disposition' /tmp/e2e-meta-headers.txt
"$PY" -c 'import json; d=json.load(open("/tmp/e2e.json")); print("json seed:", d["seed"], "canvas:", d["canvas"]["width"], "x", d["canvas"]["height"], "frames:", d["canvas"]["num_frames"])'

echo
echo "### 7. seed: two consecutive submissions differ (random mode)"
TASK2=$(curl -sS -X POST "$BASE/api/tasks" \
  -F 'mode=text' -F 'prompt=第二次随机种子测试' -F 'canvas_mode=mp' -F 'mp=0.4' \
  -F 'aspect=9:16' -F 'duration=5' -F 'budget=260' -F 'seed='"$($PY -c 'import os; print(int.from_bytes(os.urandom(8),"big") & (2**63-1))')")
SEED2=$(echo "$TASK2" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["seed"])')
ID2=$(echo "$TASK2" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "task1 seed: $SEED1"
echo "task2 seed: $SEED2"
if [ "$SEED1" != "$SEED2" ]; then echo "PASS: two random submissions used different seeds"; else echo "FAIL: seeds collided"; fi
sleep 6
curl -sS "$BASE/api/tasks/$ID2" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("task2 record seed:", d["seed"], "status:", d["status"])'

echo
echo "### 8. seed: fixed mode reuses the same seed"
FIXED=123456789012345678
TA=$(curl -sS -X POST "$BASE/api/tasks" -F 'mode=text' -F 'prompt=fixed A' -F 'canvas_mode=mp' -F 'mp=0.4' -F 'aspect=16:9' -F 'duration=5' -F 'budget=260' -F "seed=$FIXED")
TB=$(curl -sS -X POST "$BASE/api/tasks" -F 'mode=text' -F 'prompt=fixed B' -F 'canvas_mode=mp' -F 'mp=0.4' -F 'aspect=16:9' -F 'duration=5' -F 'budget=260' -F "seed=$FIXED")
SA=$(echo "$TA" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["seed"])')
SB=$(echo "$TB" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["seed"])')
echo "fixed submission A seed: $SA"
echo "fixed submission B seed: $SB"
[ "$SA" = "$FIXED" ] && [ "$SB" = "$FIXED" ] && echo "PASS: fixed mode reused $FIXED" || echo "FAIL"

echo
echo "### 9. validation: ten images / four videos / thirteen total / 1s video / no visual"
BASE="$BASE" "$PY" - <<'PY'
import io, json, os, sys, urllib.request, urllib.error
from pathlib import Path
BASE = os.environ["BASE"]
png = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                    "0000000d49444154789c6360000002000154a24f0d0000000049454e44ae426082")

def post(fields, files):
    boundary = "----e2eboundary"
    body = io.BytesIO()
    def write(text):
        body.write(text.encode())
    for key, value in fields.items():
        write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n")
    for key, name, payload in files:
        write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{name}\"\r\n")
        write("Content-Type: application/octet-stream\r\n\r\n")
        body.write(payload)
        write("\r\n")
    write(f"--{boundary}--\r\n")
    request = urllib.request.Request(BASE + "/api/tasks", data=body.getvalue(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)

base = {"mode": "references", "prompt": "limits", "canvas_mode": "mp", "mp": "0.4",
        "aspect": "16:9", "budget": "260", "low_vram": "true"}
results = []
results.append(("10 images", post(base, [("refs", f"{i}.png", png) for i in range(10)])))
results.append(("no visual (audio only)", post(base, [("audio_refs", "a.wav", b"RIFF" + b"\x00" * 60)])))
long_clip = Path("/tmp/e2e-long.mp4")
import subprocess
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2:r=24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(long_clip)], check=True)
clip = long_clip.read_bytes()
results.append(("4 videos", post(base, [("refs", "a.png", png)] + [("video_refs", f"v{i}.mp4", clip) for i in range(4)])))
results.append(("13 total", post(base, [("refs", f"{i}.png", png) for i in range(9)] +
                                 [("video_refs", f"v{i}.mp4", clip) for i in range(3)] +
                                 [("audio_refs", "a.wav", b"RIFF" + b"\x00" * 60)])))
short = Path("/tmp/e2e-short.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(short)], check=True)
results.append(("1s video", post(base, [("refs", "a.png", png), ("video_refs", "short.mp4", short.read_bytes())])))
for label, (status, payload) in results:
    print(f"  {label}: HTTP {status} -> {payload.get('error', payload)}")
PY

echo
echo "### 10. static checks"
"$PY" -m compileall -q webui && echo "compileall webui: OK"
if grep -RnE '\beval\s*\(|\bexec\s*\(|new Function\(' webui/static/*.js; then echo "FAIL: eval/exec found"; else echo "no eval/exec in JS: OK"; fi
