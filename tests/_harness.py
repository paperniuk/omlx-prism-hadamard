"""Shared setup for the adapter tests.

The tests run under oMLX's bundled interpreter (see ``run_tests.sh``) and
exercise the same code path the server uses: oMLX's pre-load patches first,
then mlx-vlm's ``load``.
"""

import importlib.util
import os
import sys

from packfinder import PACK_MODEL_TYPE, describe_search, find_packs, is_pack

MODULE_NAME = "mlx_vlm.models.prism_hadamard_qwen35"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SHIM = os.path.join(REPO_ROOT, "src", "_prism_hadamard_qwen35_shim.py")


def pack_path():
    """Pack directory: argv[1], then $OMLX_PACK, then auto-discovery.

    No models folder is hardcoded. Discovery reads oMLX's configured
    model_dirs and a few conventional locations, and identifies packs by
    their config.json, so any folder works.
    """
    explicit = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OMLX_PACK")
    if explicit:
        pack = os.path.expanduser(explicit)
        if not os.path.isdir(pack):
            raise SystemExit(f"no such directory: {pack}")
        if not is_pack(pack):
            raise SystemExit(
                f"{pack}\n  is not a {PACK_MODEL_TYPE} pack "
                "(its config.json declares a different model_type)"
            )
        return pack

    packs = find_packs()
    if not packs:
        raise SystemExit(
            f"no {PACK_MODEL_TYPE} pack found.\n{describe_search()}\n"
            "Pass one explicitly:  ./run_tests.sh tests/test_text.py /path/to/pack"
        )
    if len(packs) > 1:
        print(f"found {len(packs)} packs, using the first:")
        for pack in packs:
            print(f"  {pack}")
    return packs[0]


def apply_omlx_patches(pack, for_vlm=True):
    """Install oMLX's pre-load patches.

    These must run before mlx-vlm's model modules import: the mlx 0.32.2
    compatibility patch rewrites mlx-vlm sources as they load, and the vision
    tower crashes on ``mx.repeat`` without it.
    """
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(pack, for_vlm=for_vlm)
    if "mlx_vlm.models.qwen3_vl.vision" in sys.modules:
        raise RuntimeError(
            "mlx-vlm vision module imported before oMLX patches; fix import order"
        )


def register_shim(shim=None):
    """Register the adapter under its mlx-vlm module name.

    Mirrors what ``usercustomize.py`` does at interpreter startup, but points
    at this checkout rather than the installed copy.
    """
    shim = shim or os.environ.get("OMLX_PRISM_SHIM") or DEFAULT_SHIM
    if not os.path.exists(shim):
        raise FileNotFoundError(shim)
    import mlx_vlm.models  # noqa: F401  (parent package must exist first)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, shim)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_pack(pack, for_vlm=True):
    """Full load: patches, shim, then mlx-vlm's stock loader."""
    apply_omlx_patches(pack, for_vlm=for_vlm)
    register_shim()
    from mlx_vlm.utils import load

    return load(pack)


def chat_config(model):
    return model.config if isinstance(model.config, dict) else model.config.__dict__
