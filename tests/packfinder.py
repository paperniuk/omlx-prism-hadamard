"""Locate Prism Hadamard packs wherever they live.

Nothing about the adapter depends on a particular models folder - it is
selected by ``model_type``, not by path.  This module exists so the tests and
tools do not hardcode one either: it asks oMLX where its model directories are,
falls back to the usual suspects, and identifies packs by reading
``config.json``.
"""

import json
import os

PACK_MODEL_TYPE = "prism_hadamard_qwen35"
OMLX_SETTINGS = os.path.expanduser("~/.omlx/settings.json")
ENV_MODEL_DIRS = "OMLX_MODEL_DIRS"
FALLBACK_ROOTS = (
    "~/.lmstudio/models",
    "~/models",
    "~/.cache/huggingface/hub",
    "~/Documents/models",
)
# HF cache nests packs as models--org--name/snapshots/<hash>/config.json.
MAX_SCAN_DEPTH = 3


def omlx_model_roots():
    """Model directories configured in oMLX, newest settings format first."""
    try:
        with open(OMLX_SETTINGS) as handle:
            settings = json.load(handle)
    except (OSError, ValueError):
        return []

    model_settings = settings.get("model") or {}
    roots = list(model_settings.get("model_dirs") or [])
    single = model_settings.get("model_dir")
    if single and single not in roots:
        roots.append(single)
    return [os.path.expanduser(root) for root in roots]


def env_model_roots():
    """Extra directories from $OMLX_MODEL_DIRS (os.pathsep-separated)."""
    raw = os.environ.get(ENV_MODEL_DIRS, "")
    return [part for part in raw.split(os.pathsep) if part.strip()]


def search_roots(extra=None):
    """Every directory worth scanning, de-duplicated, existing ones only.

    Order is most-specific first: explicit argument, $OMLX_MODEL_DIRS, oMLX's
    own configuration, then conventional locations.
    """
    candidates = list(extra or [])
    candidates += env_model_roots()
    candidates += omlx_model_roots()
    candidates += [os.path.expanduser(root) for root in FALLBACK_ROOTS]

    roots, seen = [], set()
    for candidate in candidates:
        resolved = os.path.realpath(os.path.expanduser(candidate))
        if resolved in seen or not os.path.isdir(resolved):
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def is_pack(directory, model_type=PACK_MODEL_TYPE):
    """True when directory/config.json declares the given model_type."""
    config_path = os.path.join(directory, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path) as handle:
            return json.load(handle).get("model_type") == model_type
    except (OSError, ValueError):
        return False


def _scan(root, model_type, depth, visited):
    """Walk root looking for packs.

    Symlinked directories are followed - models are commonly symlinked into a
    library folder from another disk - with a visited set of real paths as the
    loop guard.
    """
    if depth < 0:
        return
    real = os.path.realpath(root)
    if real in visited:
        return
    visited.add(real)

    if is_pack(root, model_type):
        yield root
        return  # a pack contains no packs

    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():  # follows symlinks; visited guards the loops
            yield from _scan(entry.path, model_type, depth - 1, visited)


def find_packs(extra_roots=None, model_type=PACK_MODEL_TYPE):
    """All packs of the given model_type across every known models folder."""
    found, seen, visited = [], set(), set()
    for root in search_roots(extra_roots):
        for pack in _scan(root, model_type, MAX_SCAN_DEPTH, visited):
            if pack not in seen:
                seen.add(pack)
                found.append(pack)
    return found


def describe_search():
    """Human-readable account of where we looked, for error messages."""
    roots = search_roots()
    if not roots:
        return "no models directories found"
    return "searched:\n  " + "\n  ".join(roots)


if __name__ == "__main__":
    packs = find_packs()
    for pack in packs:
        print(pack)
    if not packs:
        print(f"no {PACK_MODEL_TYPE} packs found")
        print(describe_search())
