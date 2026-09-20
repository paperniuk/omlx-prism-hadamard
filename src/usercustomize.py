"""Teach mlx-vlm about Prism Hadamard Qwen3.5 packs (Ternary Bonsai 2).

mlx-vlm resolves a checkpoint's ``model_type`` to a module under
``mlx_vlm.models``.  ``prism_hadamard_qwen35`` has no such module, so oMLX
cannot load those packs.  This installs a meta-path finder that maps that one
module name to the adapter next to this file.  It is inert for every other
import and for interpreters that do not have mlx-vlm.

Remove this file and ``_prism_hadamard_qwen35_shim.py`` to undo.
"""

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "mlx_vlm.models.prism_hadamard_qwen35"
_SHIM = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_prism_hadamard_qwen35_shim.py"
)


class _PrismHadamardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or not os.path.exists(_SHIM):
            return None
        return importlib.util.spec_from_file_location(fullname, _SHIM)


if not any(type(f).__name__ == "_PrismHadamardFinder" for f in sys.meta_path):
    sys.meta_path.append(_PrismHadamardFinder())
