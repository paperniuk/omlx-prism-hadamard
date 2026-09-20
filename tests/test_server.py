"""End-to-end check against a running oMLX server.

Unlike the other tests this uses the *installed* adapter, not the checkout:
it proves the server itself can serve the pack.
"""

import base64
import io
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packfinder import PACK_MODEL_TYPE, describe_search, find_packs

BASE_URL = os.environ.get("OMLX_URL", "http://127.0.0.1:8005")
REQUEST_TIMEOUT_SECONDS = "600"


def model_name():
    """Model id: argv[1], then $OMLX_MODEL, then the discovered pack's folder.

    oMLX names a model after its directory, so the basename of any pack we can
    find is the id to ask for - whichever models folder it happens to live in.
    """
    explicit = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OMLX_MODEL")
    if explicit:
        return explicit
    packs = find_packs()
    if not packs:
        raise SystemExit(
            f"no {PACK_MODEL_TYPE} pack found.\n{describe_search()}\n"
            "Name the served model explicitly:  ./run_tests.sh tests/test_server.py <model-id>"
        )
    return os.path.basename(packs[0])


MODEL = model_name()


def post_chat(payload):
    result = subprocess.run(
        [
            "curl", "-s", "-m", REQUEST_TIMEOUT_SECONDS,
            f"{BASE_URL}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    response = json.loads(result.stdout)
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"])[:800])
    return response


def test_text():
    response = post_chat({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in exactly three words."}],
        "max_tokens": 64,
    })
    print("text ->", response["choices"][0]["message"]["content"][-200:])


def test_vision():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([90, 60, 358, 300], fill=(220, 40, 40))
    draw.rectangle([160, 330, 290, 430], fill=(30, 90, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    response = post_chat({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "What shapes and colors do you see? One short sentence."},
        ]}],
        "max_tokens": 250,
    })
    print("ground truth: red circle above a blue rectangle on white")
    print("vision ->", response["choices"][0]["message"]["content"][-300:])


def main():
    print(f"server: {BASE_URL}  model: {MODEL}")
    try:
        test_text()
        test_vision()
    except Exception as error:  # surface the server's message, do not swallow it
        print(f"FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
