#!/usr/bin/env bash
# Install the prism_hadamard_qwen35 adapter into Python's user site-packages,
# so oMLX picks it up without modifying the signed app bundle.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMLX_APP="${OMLX_APP:-/Applications/oMLX.app}"
RESOURCES="$OMLX_APP/Contents/Resources"
PYTHON_HOME="$RESOURCES/Python/cpython-3.11"
PYTHON_BIN="$PYTHON_HOME/bin/python3"
SITE_PACKAGES="$RESOURCES/Python/framework-mlx-base/lib/python3.11/site-packages"

SHIM_NAME="_prism_hadamard_qwen35_shim.py"
HOOK_NAME="usercustomize.py"
HOOK_MARKER="_PrismHadamardFinder"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$PYTHON_BIN" ] || die "oMLX's Python not found at $PYTHON_BIN (set OMLX_APP)"

run_omlx_python() {
  PYTHONHOME="$PYTHON_HOME" PYTHONPATH="$RESOURCES:$SITE_PACKAGES" \
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$@"
}

user_site="$(run_omlx_python -c 'import site; print(site.getusersitepackages())')"
enabled="$(run_omlx_python -c 'import site; print(site.ENABLE_USER_SITE)')"
[ "$enabled" = "True" ] || die "user site-packages is disabled for oMLX's interpreter"

echo "user site-packages: $user_site"
mkdir -p "$user_site"

if [ -e "$user_site/$HOOK_NAME" ] && ! grep -q "$HOOK_MARKER" "$user_site/$HOOK_NAME"; then
  die "$user_site/$HOOK_NAME already exists and is not ours.
     Merge src/$HOOK_NAME into it by hand instead of overwriting."
fi

install -m 0644 "$REPO_ROOT/src/$SHIM_NAME" "$user_site/$SHIM_NAME"
install -m 0644 "$REPO_ROOT/src/$HOOK_NAME" "$user_site/$HOOK_NAME"
echo "installed:"
echo "  $user_site/$SHIM_NAME"
echo "  $user_site/$HOOK_NAME"

echo "verifying module resolution..."
run_omlx_python -c '
import importlib, sys
assert any(type(f).__name__ == "_PrismHadamardFinder" for f in sys.meta_path), \
    "meta-path finder was not installed by usercustomize"
module = importlib.import_module("mlx_vlm.models.prism_hadamard_qwen35")
print("  resolved ->", module.__file__)
'

echo
echo "done. restart the server to pick it up:"
echo "  omlx restart"
