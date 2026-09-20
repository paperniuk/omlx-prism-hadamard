# Troubleshooting

Every error hit while building this, and what it actually meant.

### `Model type prism_hadamard_qwen35 not supported` (still, after installing)

The server has not restarted, or the hook is not loading.

```bash
omlx restart
```

Then check the finder is present in oMLX's interpreter:

```bash
R=/Applications/oMLX.app/Contents/Resources
PYTHONHOME=$R/Python/cpython-3.11 \
PYTHONPATH="$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages" \
  $R/Python/cpython-3.11/bin/python3 -c \
  "import sys; print([type(f).__name__ for f in sys.meta_path])"
```

If `_PrismHadamardFinder` is missing, user site-packages is disabled for that
interpreter (`python3 -c 'import site; print(site.ENABLE_USER_SITE)'`) - a `-s`
or `-E` flag, or `PYTHONNOUSERSITE`, will do it. In that case the only remaining
option is installing into the bundle, with the caveats in the README.

### `Model ... is unavailable after a previous load failure`

A cached failure, not a fresh one. oMLX remembers the first error and returns
409 without retrying. `omlx restart`.

### `ModuleNotFoundError: No module named 'mlx_vlm.qwen3_5'`

Relative import depth. From a module at `mlx_vlm/models/x.py`, a sibling package
is `.qwen3_5` (one dot) - `..qwen3_5` climbs out to `mlx_vlm.qwen3_5`. Note that
files *inside* `mlx_vlm/models/qwen3_5/` are one level deeper and correctly use
`..base`; do not copy their import style.

### `TypeError: repeat(): incompatible function arguments` in `qwen3_vl/vision.py`

Not your adapter. This is stock mlx-vlm against a newer mlx, and oMLX patches it
via its mlx 0.32.2 compatibility backport. The patch rewrites sources **as they
import**, so it must run before any mlx-vlm model module loads:

```python
maybe_apply_pre_load_patches(pack, for_vlm=True)   # first
register_shim()                                    # then
```

Registering the shim first pulls in the vision tower unpatched.

### `Received N parameters not in model` / `Missing N parameters`

Strict `load_weights`. Your constructed parameter tree does not match the
checkpoint. Compare them:

```python
from mlx.utils import tree_flatten
have = set(tree_flatten(model.parameters(), destination={}))
want = set(mx.load(f"{pack}/model.safetensors"))
print(sorted(want - have)[:10], sorted(have - want)[:10])
```

Usually a layer you forgot to replace, or a `signs` tensor you did not register
because the attribute was set to `None`.

### `Expected shape (a, b) but received shape (c, d)`

Placeholder geometry is wrong. For 2-bit/group-128: `weight` is
`(rows, width/16)`, `scales`/`biases` are `(rows, width/128)`, `signs` is
`(width,)`. MLX checks shapes only - dtype mismatches are fine.

### `Unsupported packed model schema` from the pack's own `runtime/artifact.py`

You are using the wrong loader for the pack. `artifact.load_model` is the
text-only, `schema_version: 1` path; a v2 vision pack must go through
`vision_artifact.load_vl_model`. (The bundled `artifact.py` cannot open its own
v2 pack - this is a bug in the pack, not in your setup.)

### `no prism_hadamard_qwen35 pack found` from a test

Discovery looked in oMLX's configured `model_dirs` and the conventional
locations and came up empty. Either point it at the folder:

```bash
OMLX_MODEL_DIRS=/Volumes/SSD/models ./run_tests.sh tests/test_text.py
```

or name the pack directly:

```bash
./run_tests.sh tests/test_text.py /Volumes/SSD/models/my-pack
```

The error prints every directory it searched. Note that discovery descends
three levels at most - deeply nested layouts need an explicit path.

### `... is not a prism_hadamard_qwen35 pack`

The directory exists but its `config.json` declares a different `model_type`.
Most often this is the **v1** pack (`Ternary-Bonsai-27B-mlx-2bit`, `model_type:
qwen3_5`), which oMLX already loads without any of this.

### The reply is all thinking and no answer

Output that reads `We need to answer user: ...` is the model's reasoning, and
seeing it as the answer means generation stopped mid-thought. These packs think
before answering, so a small `max_tokens` truncates during reasoning and
`finish_reason` comes back as `length`.

Raise `max_tokens`. oMLX separates the two correctly once generation completes:

```json
{ "reasoning_content": "We need answer just number. Compute 17*23 = 391...",
  "content": "391" }
```

For shorter deliberation send `"reasoning_effort": "medium"` - the chat template
accepts `xhigh` (default), `medium` and `low`, and oMLX normalizes other
OpenAI-style values onto those. The pack's README notes `low` behaves close to
`xhigh`, so `medium` is the only real speed lever.

### Transformers warnings on load

```
You are using a model of type `prism_hadamard_qwen35` to instantiate a model of type ``
The tokenizer you are loading ... with an incorrect regex pattern
```

Both are expected and harmless. The first is transformers noticing the
deliberately non-standard `model_type`; the second comes from the pack's
tokenizer, which was copied from the source GGUF.

### After an oMLX update

The adapter lives in your home directory and survives updates - but mlx-vlm's
version may move under it. Re-run the tests after updating:

```bash
./run_tests.sh tests/test_text.py
```

If `get_model_and_args` changed, re-read it (see
[adapting.md](adapting.md#step-1--find-the-dispatch-point)).
