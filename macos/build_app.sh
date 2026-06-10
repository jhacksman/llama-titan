#!/bin/bash
# Assemble dist/Madrone Way.app from the repo sources. Runs anywhere
# (no Xcode needed) — the bundle is scripts + resources, nothing compiled.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Madrone Way.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$ROOT/macos/Info.plist" "$APP/Contents/"
printf 'APPL????' > "$APP/Contents/PkgInfo"

cp "$ROOT/macos/launcher.sh" "$APP/Contents/MacOS/MadroneWay"
chmod +x "$APP/Contents/MacOS/MadroneWay"

cp -R "$ROOT/app" "$APP/Contents/Resources/app"
find "$APP/Contents/Resources/app" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp "$ROOT/macos/desktop.py" "$ROOT/macos/requirements-mac.txt" "$APP/Contents/Resources/"

python3 "$ROOT/macos/make_icon.py" "$APP/Contents/Resources/icon.icns"

echo "built: $APP"
