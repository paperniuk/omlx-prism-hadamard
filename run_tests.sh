#!/usr/bin/env bash
# Run a test under oMLX's bundled interpreter.
#   ./run_tests.sh                      # text test, default pack
#   ./run_tests.sh tests/test_vision.py # a specific test
#   ./run_tests.sh tests/test_text.py ~/models/some-pack
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMLX_APP="${OMLX_APP:-/Applications/oMLX.app}"
RESOURCES="$OMLX_APP/Contents/Resources"
PYTHON_HOME="$RESOURCES/Python/cpython-3.11"
PYTHON_BIN="$PYTHON_HOME/bin/python3"
SITE_PACKAGES="$RESOURCES/Python/framework-mlx-base/lib/python3.11/site-packages"

[ -x "$PYTHON_BIN" ] || { echo "error: oMLX's Python not found at $PYTHON_BIN" >&2; exit 1; }

test_file="${1:-$REPO_ROOT/tests/test_text.py}"
shift || true

PYTHONHOME="$PYTHON_HOME" \
PYTHONPATH="$RESOURCES:$SITE_PACKAGES" \
PYTHONDONTWRITEBYTECODE=1 \
  exec "$PYTHON_BIN" "$test_file" "$@"
