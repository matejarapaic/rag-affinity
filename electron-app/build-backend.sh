#!/usr/bin/env bash
# build-backend.sh — Bundle the Python backend with PyInstaller
#
# Usage (run from the project root /Users/matejarapaic/RAG/):
#   bash electron-app/build-backend.sh
#
# Or from inside electron-app/:
#   bash build-backend.sh
#
# After this script completes, electron-app/resources/backend/ will contain
# the compiled binary. Then run: npm run dist  (from electron-app/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Project root: $PROJECT_ROOT"
echo "==> Spec file:    $SCRIPT_DIR/backend.spec"

# Prefer the project venv if it exists
VENV_PYTHON="$PROJECT_ROOT/rag-chatbot/backend/.venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
  PYTHON="$VENV_PYTHON"
  echo "==> Using venv Python: $PYTHON"
else
  PYTHON="$(which python3)"
  echo "==> Using system Python: $PYTHON"
fi

# Ensure PyInstaller is available
if ! "$PYTHON" -m PyInstaller --version &>/dev/null; then
  echo "==> Installing PyInstaller..."
  "$PYTHON" -m pip install pyinstaller
fi

# Clean previous build artifacts
rm -rf "$SCRIPT_DIR/resources/backend"
rm -rf "$PROJECT_ROOT/build/backend"

echo "==> Running PyInstaller..."
"$PYTHON" -m PyInstaller \
  --distpath "$SCRIPT_DIR/resources" \
  --workpath "$PROJECT_ROOT/build" \
  --noconfirm \
  "$SCRIPT_DIR/backend.spec"

echo ""
echo "==> Build complete!"
echo "    Binary output: $SCRIPT_DIR/resources/backend/"
echo ""
echo "    Next step: cd electron-app && npm run dist"
