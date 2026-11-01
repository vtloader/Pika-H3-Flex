#!/usr/bin/env bash
# Launch the Pika H3 Flex browser WebUI with the NPU virtualenv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="$ROOT/.venv312/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$VENV_PYTHON" ]]; then
  PYTHON="$VENV_PYTHON"
else
  PYTHON="$(command -v python3)"
  echo "warning: .venv312 not found; using $PYTHON" >&2
fi

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-3000}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PIKA_CPU_THREADS="${PIKA_CPU_THREADS:-16}"

echo "Pika H3 Flex WebUI: http://$HOST:$PORT"
exec "$PYTHON" -u -m webui.server --host "$HOST" --port "$PORT"
