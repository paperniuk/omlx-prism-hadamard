"""Load a prism_hadamard_qwen35 pack and describe a generated test image."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _harness import chat_config, load_pack, pack_path

MAX_TOKENS = 80
PROMPT = "Describe the shapes and their colors."
GROUND_TRUTH = "red circle above a blue rectangle, white background"
IMAGE_PATH = "/tmp/prism_shape_test.png"


def draw_test_image(path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([90, 60, 358, 300], fill=(220, 40, 40))
    draw.rectangle([160, 330, 290, 430], fill=(30, 90, 200))
    image.save(path)
    return path


def main():
    pack = pack_path()
    print(f"pack: {pack}")
    print(f"ground truth: {GROUND_TRUTH}")

    model, processor = load_pack(pack)
    image = draw_test_image(IMAGE_PATH)

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, chat_config(model), PROMPT, num_images=1)
    start = time.time()
    out = generate(
        model, processor, prompt, image=[image], max_tokens=MAX_TOKENS, verbose=False
    )
    print("---- output ----")
    print(getattr(out, "text", out))
    print(f"---- {time.time() - start:.1f}s ----")


if __name__ == "__main__":
    main()
