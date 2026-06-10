#!/bin/bash
# Contents/MacOS launcher for Madrone Way.app.
# First run: creates a private Python env under ~/Library/Application Support
# and installs the (keyless, pure-pip) dependencies. Later runs start instantly.

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
SUP="$HOME/Library/Application Support/MadroneDashboard"
VENV="$SUP/venv"
LOG="$SUP/launch.log"

mkdir -p "$SUP"
exec >>"$LOG" 2>&1
echo "=== launch $(date)"

notify() {
  /usr/bin/osascript -e "display notification \"$1\" with title \"Madrone Way\"" || true
}

PY="$(command -v python3 || echo /usr/bin/python3)"
if ! "$PY" -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
  /usr/bin/osascript -e 'display alert "Madrone Way needs Python 3" message "Install the Apple Command Line Tools (run: xcode-select --install in Terminal) or python.org Python 3, then open the app again."'
  exit 1
fi

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import fastapi, uvicorn, httpx' 2>/dev/null; then
  notify "First launch: setting up (about a minute)…"
  rm -rf "$VENV"
  "$PY" -m venv "$VENV" || exit 1
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$RES/requirements-mac.txt" \
    || "$VENV/bin/pip" install --quiet fastapi "uvicorn[standard]" httpx  # window lib optional
  notify "Setup done — opening the dashboard."
fi

cd "$RES"
exec "$VENV/bin/python" desktop.py
