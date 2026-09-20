# omlx-prism-hadamard

**Ternary Bonsai 2 doesn't run in oMLX. This makes it run.**

## What this is about

Prism ML publishes `Ternary-Bonsai-2-27B-mlx-2bit` - a 27-billion-parameter
model that reads images as well as text, compressed to roughly **2 bits per
weight**. That is 8.4 GB on disk instead of the ~54 GB the same model needs at
full precision. On a Mac that is the difference between "this model does not
fit" and "this model runs, with room to spare".

So you download it into whichever folder oMLX scans for models, open oMLX, and
everything looks right. oMLX finds it at startup:

```
omlx.model_discovery - Discovered model: Ternary-Bonsai-2-27B-mlx-2bit
                       (type: vlm, engine: vlm, size: 8.41GB, text-only: 7.50GB)
```

It appears in the model list. It appears in `GET /v1/models`. The size is right,
the engine is right. Nothing warns you.

Then you send it a message:

```
POST /v1/chat/completions → 409
Model 'Ternary-Bonsai-2-27B-mlx-2bit' is unavailable after a previous load failure:
VLM load failed: Model type prism_hadamard_qwen35 not supported.
Error: No module named 'mlx_vlm.speculative.drafters.prism_hadamard_qwen35';
LLM fallback also failed: Model type prism_hadamard_qwen35 not supported.
```

And it stays broken. oMLX caches that first failure and returns 409 for every
later request **without retrying** - so the model sits in your list looking
perfectly available and answers nothing. Clicking around the UI does not help,
because nothing in the UI is wrong.

### Why it happens, in one paragraph

This model is not stored the way normal models are. Squeezing 27B parameters
down to 2 bits without destroying the model required rotating the weights into a
different mathematical basis first (a Hadamard transform - that is where this
repo's name comes from). The code that undoes that rotation at inference time
ships *inside the model folder*, in a `runtime/` directory that no standard
loader knows to look at. The author even labelled the pack so that no standard
loader would try: `model_type` is `prism_hadamard_qwen35`, a name nothing
recognises. That is a deliberate guardrail, not a bug - feeding these weights to
an ordinary loader would produce confident nonsense, which is worse than an
error.

### What this repo does

It supplies the one thing that is missing: a loader registered under that name,
which applies the rotation exactly the way the pack's own runtime does. After
installing, the model loads into oMLX's normal VLM engine and works - text
**and** vision - with oMLX's full Qwen3.5 optimisation stack (FA-256 steel
attention, GDN prefill kernels, ragged decode) still applied.

It also doubles as a worked example of **how to add any unrecognised
architecture to mlx-vlm / oMLX**, because the technique is general - see
[docs/adapting.md](docs/adapting.md).

---

## Why it fails (the mechanics)

mlx-vlm resolves a checkpoint's `model_type` by *importing a module of that
name*:

```python
# mlx_vlm/utils.py - get_model_and_args()
for pkg in ("mlx_vlm.models", "mlx_vlm.speculative.drafters"):
    arch = importlib.import_module(f"{pkg}.{model_type}")
```

That import is the whole gate, and `prism_hadamard_qwen35` is not an
architecture anyone ships - so it throws, the LLM fallback throws for the same
reason, and oMLX gives up. The pack says as much itself, in `PACK-RUNTIME.md`:

> Ordinary MLX loaders do not apply the required transforms.

Worth being precise about the failure mode: loading this pack as a plain
`qwen3_5` checkpoint would not merely be slow or lossy. The tensors are packed
`uint32` with separate sign vectors, so the strict weight load rejects them
outright - and if you forced it past that, you would get fluent garbage.

## How the fix works

Supply the missing module. It turns out the pack is *exactly* mlx-vlm's
`qwen3_5` graph with the language-model projections swapped for Hadamard-aware
2-bit layers, so the adapter is thin:

```python
class Model(_Qwen35Model):
    def __init__(self, config):
        super().__init__(config)
        install_packed_modules(self.language_model, config.packed_modules)
```

`fwht` and `Packed` are ported verbatim from the pack's own `runtime/runtime.py`
so the numerics are identical to the reference loader. Full walkthrough:
[docs/how-it-works.md](docs/how-it-works.md).

## Install

```bash
git clone https://github.com/paperniuk/omlx-prism-hadamard.git
cd omlx-prism-hadamard
./install.sh
omlx restart
```

`install.sh` copies two files into Python's **user** site-packages
(`~/.local/lib/python3.11/site-packages/`) and verifies the module resolves:

| File                               | Role                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `usercustomize.py`               | Meta-path finder, auto-imported at interpreter startup. Reacts to exactly one module name. |
| `_prism_hadamard_qwen35_shim.py` | The adapter itself.                                                                        |

Nothing inside `/Applications/oMLX.app` is touched - see
[Why not patch the app bundle](#why-not-patch-the-app-bundle).

### Verify

```bash
./run_tests.sh tests/test_text.py     # loads the pack directly, generates text
./run_tests.sh tests/test_vision.py   # draws a test image, asks the model about it
./run_tests.sh tests/test_server.py   # hits a running oMLX server over HTTP
```

The tests find the pack themselves - no models folder is hardcoded anywhere.
They read oMLX's configured `model_dirs`, plus a few conventional locations, and
identify packs by reading each `config.json` for `model_type`. Name one
explicitly whenever you prefer:

```bash
./run_tests.sh tests/test_text.py /Volumes/SSD/models/Ternary-Bonsai-2-27B-mlx-2bit
OMLX_PACK=~/whatever/bonsai2 ./run_tests.sh tests/test_vision.py
```

To make discovery look somewhere unusual, point it there:

```bash
OMLX_MODEL_DIRS=/Volumes/SSD/models:/data/packs ./run_tests.sh tests/test_text.py
```

`tests/packfinder.py` also runs standalone and just lists what it found:

```bash
python3 tests/packfinder.py
```

### Which folder should the model go in?

Any folder oMLX is configured to scan - this adapter never looks at the path.
It is selected by the checkpoint's `model_type`, so a pack works the same from
your home directory, an external SSD, or anywhere else.

oMLX keeps its list in `~/.omlx/settings.json`:

```json
{ "model": { "model_dirs": ["/Users/you/models"] } }
```

Add folders through the oMLX app's settings (or edit that list and restart with
`omlx restart`).

### Uninstall

```bash
./uninstall.sh
omlx restart
```

## Verified

On an M1 Max / 64 GB, oMLX 0.7.0.dev2, pack `Ternary-Bonsai-2-27B-mlx-2bit`:

|               | Result                                                                               |
| ------------- | ------------------------------------------------------------------------------------ |
| Load          | 3.4 s, 8.38 GB resident                                                              |
| Text          | coherent output via`/v1/chat/completions`                                          |
| Vision        | red circle + blue rectangle test image →*"I see a red circle and a blue square."* |
| Decode (warm) | 18.1 tok/s                                                                           |

For reference the v1 pack (`Ternary-Bonsai-27B-mlx-2bit`, stock `qwen3_5`
`model_type`) runs 25.3 tok/s on the same machine. See
[Performance](#performance).

## Why not patch the app bundle

The obvious fix is dropping the adapter into
`oMLX.app/Contents/Resources/Python/.../mlx_vlm/models/`. Don't:

- The app is signed with a hardened runtime and **sealed resources**
  (`Sealed Resources version=2`), and still carries a quarantine xattr. Adding a
  file breaks the seal and risks a "damaged app" prompt on next launch.
- Every oMLX update replaces `Resources/` and silently drops your fix.

Python's `usercustomize` hook avoids both. It is imported by `site.py` at
startup for any interpreter with user site enabled - including the one oMLX's
GUI spawns - and it lives in your home directory, so updates leave it alone.

The tradeoff: the hook is loaded by *every* Python 3.11 on the machine that has
user site enabled. It is written to be inert - it installs one meta-path finder
that returns `None` for every name but
`mlx_vlm.models.prism_hadamard_qwen35`.

## Performance

18.1 tok/s vs 25.3 tok/s for the v1 pack. Two likely contributors:

1. **The Hadamard transform itself** - one `mx.hadamard_transform` per packed
   projection, on top of the matmul.
2. **`Packed` bypasses oMLX's tuned 2-bit path.** oMLX's `bonsai_qmv` patch
   targets `nn.QuantizedLinear`; `Packed` calls `mx.quantized_matmul` directly,
   so it may not benefit.

Recovering (2) means teaching `bonsai_qmv` about `Packed`, or reshaping `Packed`
as a `QuantizedLinear` subclass that pre-rotates its input. Untried -
contributions welcome.

## Scope

- **oMLX only.** Other runners have their own loaders; this does not make the
  pack work in them.
- Verified against oMLX 0.7.0.dev2 with mlx-vlm 0.6.3 / mlx 0.32.x. The
  `model_type` dispatch in `get_model_and_args` has been stable for a long time,
  but check [docs/troubleshooting.md](docs/troubleshooting.md) if a future
  release breaks it.
- MTP is not carried by these packs (`components.mtp: false`), so nothing here
  touches speculative decoding.

## Docs

- [docs/how-it-works.md](docs/how-it-works.md) - pack anatomy, the dispatch
  path, what `Packed` computes, why strict `load_weights` succeeds.
- [docs/adapting.md](docs/adapting.md) - the general recipe for an unsupported
  `model_type`, and how to investigate one yourself.
- [docs/troubleshooting.md](docs/troubleshooting.md) - every error hit while
  building this, and what it meant.

## License

MIT. `fwht` and `Packed` in the adapter are ported from the pack's bundled
`runtime/runtime.py` (MIT, Copyright © 2023 Apple Inc.) - see [NOTICE.md](NOTICE.md).
