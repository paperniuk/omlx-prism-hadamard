#!/usr/bin/env bash
# Remove the prism_hadamard_qwen35 adapter from user site-packages.
set -euo pipefail

OMLX_APP="${OMLX_APP:-/Applications/oMLX.app}"
RESOURCES="$OMLX_APP/Contents/Resources"
PYTHON_HOME="$RESOURCES/Python/cpython-3.11"
PYTHON_BIN="$PYTHON_HOME/bin/python3"
SITE_PACKAGES="$RESOURCES/Python/framework-mlx-base/lib/python3.11/site-packages"

SHIM_NAME="_prism_hadamard_qwen35_shim.py"
HOOK_NAME="usercustomize.py"
HOOK_MARKER="_PrismHadamardFinder"

[ -x "$PYTHON_BIN" ] || { echo "error: oMLX's Python not found at $PYTHON_BIN" >&2; exit 1; }

user_site="$(PYTHONHOME="$PYTHON_HOME" PYTHONPATH="$RESOURCES:$SITE_PACKAGES" \
  "$PYTHON_BIN" -c 'import site; print(site.getusersitepackages())')"

rm -f "$user_site/$SHIM_NAME"
if [ -e "$user_site/$HOOK_NAME" ] && grep -q "$HOOK_MARKER" "$user_site/$HOOK_NAME"; then
  rm -f "$user_site/$HOOK_NAME"
else
  echo "left $user_site/$HOOK_NAME alone (not ours)"
fi

echo "removed. restart the server: omlx restart"
