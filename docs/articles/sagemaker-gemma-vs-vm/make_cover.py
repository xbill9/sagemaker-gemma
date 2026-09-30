"""Generate this article's dev.to cover with Gemini, crop it to dev.to's displayed band
(1376x578, 2.381:1) and name it by its bytes. Writes evidence/cover-provenance.txt.

    python3 make_cover.py [--top N]      # N = crop offset in the 1376x768 render

The key is read from ~/gemini.key, never from the command line.
"""

import hashlib
import io
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

MODEL = "gemini-3.1-flash-lite-image"
HERE = Path(__file__).resolve().parent
PROMPT = (
    "Wide banner illustration for a technical blog cover, dark charcoal background with a faint blueprint grid. "
    'Bold clean white sans-serif headline centered near the top: "Gemma 4: SageMaker or a VM?". '
    "Below it, a polished isometric 3D scene in the middle of the frame: on the left, a glowing orange cloud-shaped "
    "managed service platform with a small server inside it; on the right, a single bare server box holding a small "
    "low-profile single-slot NVIDIA T4 GPU card with a green edge; a bright blue light trail flows from the cloud "
    "platform to the bare server. Minimal, modern, high contrast, blue and orange accents, no clutter, no gauges, "
    "no numbers, no other text, no logos, no letterboxing. Keep all important content inside the central horizontal "
    "band, with generous empty margin at the top and bottom edges."
)


def main() -> None:
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 95
    key = Path("~/gemini.key").expanduser().read_text().strip()
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="16:9")
        ),
    )
    data = next(p.inline_data.data for p in resp.candidates[0].content.parts if p.inline_data)
    raw = Image.open(io.BytesIO(data)).convert("RGB")
    raw.save(HERE / "cover-raw.png")
    img = raw.resize((1376, 768), Image.LANCZOS) if raw.size != (1376, 768) else raw
    box = (0, top, 1376, top + 578)
    out = io.BytesIO()
    img.crop(box).save(out, "JPEG", quality=90)
    name = f"devto-cover.{hashlib.sha256(out.getvalue()).hexdigest()[:8]}.jpg"
    (HERE / name).write_bytes(out.getvalue())
    os.makedirs(HERE / "evidence", exist_ok=True)
    (HERE / "evidence" / "cover-provenance.txt").write_text(
        f"# cover provenance -- {name}\n"
        f"# generated with {MODEL} via google-genai models.generate_content, aspect 16:9\n"
        f"# raw {raw.size[0]}x{raw.size[1]} -> 1376x768 -> crop box {box} -> 1376x578 (2.381:1) -> JPEG q90\n"
        f"# name = sha256(bytes)[:8]\n# prompt:\n# {PROMPT}\n"
    )
    print(name, raw.size)


if __name__ == "__main__":
    main()
