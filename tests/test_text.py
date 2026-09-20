"""Load a prism_hadamard_qwen35 pack and generate text."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _harness import chat_config, load_pack, pack_path

MAX_TOKENS = 60
PROMPT = "In one sentence: why is the sky blue?"


def main():
    pack = pack_path()
    print(f"pack: {pack}")

    start = time.time()
    model, processor = load_pack(pack)
    print(f"loaded in {time.time() - start:.1f}s -> {type(model).__module__}")

    import mlx.core as mx

    print(f"peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, chat_config(model), PROMPT, num_images=0)
    start = time.time()
    out = generate(model, processor, prompt, max_tokens=MAX_TOKENS, verbose=False)
    print("---- output ----")
    print(getattr(out, "text", out))
    print(f"---- {time.time() - start:.1f}s ----")


if __name__ == "__main__":
    main()
