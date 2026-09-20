# How it works

## 1. The pack

`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` is not a normal MLX checkpoint. Its
`config.json`:

```json
{
  "schema_version": 2,
  "model_type": "prism_hadamard_qwen35",
  "base_model_type": "qwen3_5",
  "tensor_namespace": "mlx-vlm-qwen3_5",
  "requires_runtime": "runtime/artifact.py",
  "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
  "components": {"text": true, "vision": true, "mtp": false},
  "modules": [ {"path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"}, ... ]
}
```

Three things matter:

- **`tensor_namespace: mlx-vlm-qwen3_5`** - the weights are already named for
  mlx-vlm's `qwen3_5` module tree (`language_model.model.layers.N...`). The
  graph is stock; only the storage is exotic.
- **`modules`** - 402 records naming every layer stored in the rotated 2-bit
  basis. One is the embedding (`model.embed_tokens`), the rest are projections
  including `lm_head`. All use `block: 1024`.
- **`components.vision: true`** - the vision tower is present, and it is plain
  unquantized fp16. Only the *language model* is packed.

Per-tensor storage for a packed module of shape `(rows, width)`:

| Tensor | dtype | shape |
|---|---|---|
| `weight` | `uint32` | `(rows, width / 16)` - 16 two-bit values per word |
| `scales` | `float16` | `(rows, width / 128)` |
| `biases` | `float16` | `(rows, width / 128)` |
| `signs` | `float32` | `(width,)` - ±1, only when `block != 0` |

## 2. What `Packed` computes

The weights live in a rotated basis, so activations must be rotated to match
before the matmul:

```python
def fwht(x, block, signs, inverse=False):
    if not inverse:
        x = x * signs                      # sign flip, then
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1/math.sqrt(block))
    if inverse:
        x = x * signs                      # ...or transform, then sign flip
    return x
```

A projection rotates its input and then does an ordinary 2-bit quantized
matmul:

```python
x = fwht(x, self.block, self.signs)
return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                           transpose=True, group_size=128, bits=2)
```

The embedding goes the other way - it gathers and dequantizes rows, then
applies the **inverse** transform to bring the result back into the model's
normal basis.

Skip the transform and the matmul is still shape-valid but numerically
meaningless. That is why the pack ships its own loader and refuses to name a
known architecture.

## 3. The dispatch point

`mlx_vlm/utils.py`:

```python
def get_model_and_args(config: dict):
    model_type = config["model_type"].lower()
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    for pkg in ("mlx_vlm.models", "mlx_vlm.speculative.drafters"):
        try:
            return importlib.import_module(f"{pkg}.{model_type}"), model_type
        except ImportError as e:
            ...
    raise ValueError(f"Model type {model_type} not supported. Error: {last_err}")
```

That is the whole gate. `importlib.import_module` consults `sys.modules` and
`sys.meta_path` first, so a module registered under the right name is
indistinguishable from one shipped inside the package. No patching of mlx-vlm
needed.

The module must expose what `load_model` reaches for:

- `ModelConfig.from_dict(config)`
- `TextConfig` / `VisionConfig` (used by `update_module_configs`)
- `Model(model_config)`
- `LanguageModel` / `VisionModel` (used by `sanitize_weights`)

## 4. The adapter

Three pieces.

**Config.** `ModelConfig.from_dict` filters unknown keys, so the pack's
`modules` list would be dropped. A subclass keeps it under a non-colliding
name:

```python
@dataclass
class ModelConfig(_Qwen35ModelConfig):
    packed_modules: List[Dict[str, Any]] = field(default_factory=list)
    base_model_type: str = "qwen3_5"
    components: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params.setdefault("packed_modules", params.get("modules") or [])
        return super().from_dict(params)
```

**Model.** Build the stock graph, then replace the packed submodules:

```python
class Model(_Qwen35Model):
    def __init__(self, config):
        super().__init__(config)
        install_packed_modules(self.language_model, config.packed_modules)
```

**Placeholders.** This is the subtle part. mlx-vlm calls
`model.load_weights(..., strict=True)` *after* construction, and strict mode
demands that the parameter tree match the checkpoint exactly - same keys, same
shapes. So `install_packed_modules` does not load any data; it installs
correctly shaped **empty** `Packed` layers, derived from the original module's
dimensions:

```python
rows, width = original.weight.shape
arrays = (
    mx.zeros((rows, width // 16), dtype=mx.uint32),    # 2-bit words
    mx.zeros((rows, width // 128), dtype=mx.float16),  # scales
    mx.zeros((rows, width // 128), dtype=mx.float16),  # biases
)
signs = mx.ones((width,), dtype=mx.float32) if block else None
```

`load_weights` then fills them. MLX's strict check compares **shapes only**, not
dtypes, so the placeholder dtypes just need to be sane.

## 5. Why mlx-vlm's quantization step leaves it alone

`load_model` sees `"quantization"` in the config and calls `nn.quantize(...)`.
It is harmless here because of its own predicate:

```python
if not hasattr(m, "to_quantized"):
    return False          # Packed has no to_quantized  -> skipped
return f"{p}.scales" in weights
                          # vision tower Linears have no .scales -> skipped
```

So every packed layer is skipped as unquantizable, and every unpacked layer is
skipped as unquantized. Nothing is touched.

## 6. Prompt formatting and the processor

Two registrations make vision work:

```python
prompt_utils.MODEL_CONFIG.setdefault(PACK_MODEL_TYPE, MODEL_CONFIG["qwen3_5"])
install_auto_processor_patch(PACK_MODEL_TYPE, Qwen3VLProcessor)
```

Without the first, `apply_chat_template` falls through to the "unknown model
type" branch and formats text-only messages - images are silently dropped
rather than raising. Without the second, processor resolution goes through
`AutoProcessor` on an unknown `model_type`.

## 7. The startup hook

`usercustomize.py` is imported by `site.py` at interpreter startup whenever user
site-packages is enabled. It installs a meta-path finder that answers for one
name:

```python
class _PrismHadamardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or not os.path.exists(_SHIM):
            return None
        return importlib.util.spec_from_file_location(fullname, _SHIM)
```

`spec_from_file_location` with a dotted name sets the module's `__package__` to
`mlx_vlm.models`, so the adapter's relative imports (`from .qwen3_5 import ...`)
resolve exactly as if the file lived inside the package.

The finder is *appended* to `sys.meta_path`, so it is consulted only after the
normal path finder has failed - it can never shadow a real module.

## 8. Load order matters

oMLX's `maybe_apply_pre_load_patches()` installs an import hook that rewrites
mlx-vlm sources as they load (the mlx 0.32.2 compatibility backport). It must
run **before** any mlx-vlm model module imports.

In the server this is automatic: patches run, then `get_model_and_args` imports
the adapter, which imports `qwen3_5`, which is rewritten on the way in.

In a standalone script it is easy to get wrong - importing the adapter first
pulls in the vision tower unpatched, and you get:

```
TypeError: repeat(): incompatible function arguments.
  Invoked with types: mlx.core.array, mlx.core.array
```

`tests/_harness.py` enforces the order and asserts the vision module has not
been imported early.
