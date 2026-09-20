# Adapting this for another checkpoint

The general problem: oMLX (or plain mlx-vlm / mlx-lm) refuses a checkpoint with
`Model type <x> not supported`. This is the recipe used to fix it once.

## Step 0 - read the checkpoint, not the error

```bash
PACK=/path/to/your/pack          # any folder; python3 tests/packfinder.py lists known ones
python3 -c "
import json; c=json.load(open('$PACK/config.json'))
print([k for k in c if k != 'text_config'])
print('model_type:', c.get('model_type'))
print('base/arch :', c.get('base_model_type'), c.get('architectures'))
"
ls "$PACK"
```

Look for:

- **`base_model_type` / `architectures` / `tensor_namespace`** - does the pack
  admit what graph it really is? If so, most of your work is done.
- **A bundled loader** (`runtime/`, `*_artifact.py`, `PACK-RUNTIME.md`,
  `requires_runtime`). Read it. It is the ground truth for the math, and often
  reveals the checkpoint is a stock architecture with custom storage.
- **A `modules` / layer-manifest list** - tells you exactly which layers deviate.

For this pack, `runtime/vision_artifact.py` gave the whole game away in ten
lines: build `mlx_vlm.models.qwen3_5`, swap some submodules, load strictly.

## Step 1 - find the dispatch point

mlx-vlm and mlx-lm both map `model_type` to a module import:

```bash
S=/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages
grep -n "not supported" -B 25 $S/mlx_vlm/utils.py | head -40
```

Note `MODEL_REMAPPING` while you are there. If your checkpoint is a plain alias
of a supported architecture - same math, different name - a one-line remap
entry is the entire fix and you can stop here.

> Only take the remap shortcut when the *numerics* are unchanged. If the
> checkpoint stores weights in a transformed basis, aliasing it produces a model
> that loads and talks nonsense.

## Step 2 - write the module

Minimum surface `load_model` touches:

| Symbol | Why |
|---|---|
| `ModelConfig.from_dict` | builds the config; subclass the base architecture's and re-add any custom keys it would filter out |
| `TextConfig`, `VisionConfig` | `update_module_configs` reads them |
| `Model` | the graph |
| `LanguageModel`, `VisionModel` | `sanitize_weights` looks for them |

Subclass the closest stock architecture rather than reimplementing it. Port
custom layers from the pack's own runtime verbatim - matching the reference
numerics matters more than elegance, and it keeps the license story simple.

If you install custom layers, remember `load_weights(strict=True)` runs
afterwards: build **correctly shaped empty** layers at construction time and let
the loader fill them. Derive shapes from the module you are replacing, never
hardcode them.

## Step 3 - check you are not fighting the quantizer

`load_model` calls `nn.quantize()` whenever `config["quantization"]` exists.
Make sure its predicate skips your layers - custom modules without a
`to_quantized` method are skipped automatically, which is usually what you want.

## Step 4 - register it without touching the app bundle

Copy `src/usercustomize.py` and change `_TARGET` and `_SHIM`. Everything else
carries over. To support several adapters, keep a dict of
`{module_name: file_path}` and look up `fullname` in it.

Install into user site-packages (`./install.sh` shows the path computation) and
verify:

```bash
./run_tests.sh tests/test_text.py "$PACK"
```

## Step 5 - vision, if the pack has a tower

Register the prompt format and processor, or images are silently dropped:

```python
prompt_utils.MODEL_CONFIG.setdefault(MY_TYPE, prompt_utils.MODEL_CONFIG["<base>"])
install_auto_processor_patch(MY_TYPE, <BaseProcessor>)
```

## Debugging notes

**Always apply oMLX's pre-load patches before importing mlx-vlm model
modules** in standalone scripts:

```python
from omlx.utils.model_loading import maybe_apply_pre_load_patches
maybe_apply_pre_load_patches(pack, for_vlm=True)
```

They are what makes mlx-vlm work against the mlx version oMLX ships. Skipping
them produces confusing failures deep inside stock mlx-vlm code that have
nothing to do with your adapter.

**Run everything under oMLX's interpreter**, not the system one:

```bash
R=/Applications/oMLX.app/Contents/Resources
PYTHONHOME=$R/Python/cpython-3.11 \
PYTHONPATH="$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages" \
  $R/Python/cpython-3.11/bin/python3 your_script.py
```

`run_tests.sh` is a wrapper for exactly this.

**Watch the server log** - it names every patch that fires, which tells you
whether your model is getting the fast paths:

```bash
tail -f ~/.omlx/logs/server.log
```

**A failed load is cached.** oMLX will not retry until models are rescanned;
`omlx restart` is the reliable reset.
