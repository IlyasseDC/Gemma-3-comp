"""
Inspection tool for Gemma 4 E4B image-to-text outputs.

Saves to inspect_outputs/:
  images/sample_000.jpg  ...  — the raw COCO images
  captions.json              — generated + reference captions per sample
  report.html                — browser-viewable side-by-side comparison

Usage:
    python3 inspect_outputs.py --n_samples 20
"""

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",       default="google/gemma-4-E4B")
    p.add_argument("--n_samples",      type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--prompt",         default="Describe this image in one sentence.")
    p.add_argument("--out_dir",        default="inspect_outputs")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(model_id, dtype_str):
    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Loading model ({dtype_str}) …")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=DTYPE_MAP[dtype_str], device_map="auto"
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_caption(model, processor, image, prompt, max_new_tokens, device):
    try:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ValueError, AttributeError):
        image_token = getattr(processor, "image_token", "<|image|>")
        text = f"{image_token}\n{prompt}"

    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ── Dataset ──────────────────────────────────────────────────────────────────

def fetch_samples(n_samples):
    ds = load_dataset("yerevann/coco-karpathy", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    session = requests.Session()
    samples = []
    for item in tqdm(ds, desc="Fetching images", leave=False):
        try:
            resp = session.get(item["url"], timeout=15)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"  Skip {item['url']}: {e}")
            continue
        captions = item["sentences"]
        if not isinstance(captions, list):
            captions = [captions]
        samples.append({
            "coco_id":   item["cocoid"],
            "url":       item["url"],
            "image":     image,
            "refs":      captions,
        })
    return samples


# ── Image → base64 for HTML ──────────────────────────────────────────────────

def img_to_b64(image: Image.Image, max_side=400) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML report ──────────────────────────────────────────────────────────────

CARD_STYLE = """
body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 20px; }
h1   { color: #333; }
.grid { display: flex; flex-wrap: wrap; gap: 20px; }
.card { background: white; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.15);
        padding: 14px; width: 420px; }
.card img { width: 100%; border-radius: 4px; }
.label { font-size: 11px; font-weight: bold; color: #888; margin: 8px 0 2px; text-transform: uppercase; }
.gen  { background: #eef6ff; border-left: 3px solid #3a8dff; padding: 6px 10px;
        border-radius: 4px; font-size: 13px; margin-bottom: 6px; }
.ref  { background: #f0fff4; border-left: 3px solid #38c172; padding: 6px 10px;
        border-radius: 4px; font-size: 12px; color: #444; margin-bottom: 4px; }
.meta { font-size: 11px; color: #aaa; margin-top: 8px; }
"""

def build_html(records, prompt, model_id):
    cards = ""
    for i, r in enumerate(records):
        b64 = img_to_b64(r["image"])
        refs_html = "".join(f'<div class="ref">{ref.strip()}</div>' for ref in r["refs"])
        cards += f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{b64}" />
          <div class="label">Generated caption</div>
          <div class="gen">{r["generated"] or "<em>(empty)</em>"}</div>
          <div class="label">Reference captions ({len(r["refs"])})</div>
          {refs_html}
          <div class="meta">COCO id {r["coco_id"]} &nbsp;|&nbsp; sample {i}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Gemma 4 E4B — Inspection</title>
<style>{CARD_STYLE}</style>
</head><body>
<h1>Gemma 4 E4B — Caption Inspection</h1>
<p><b>Model:</b> {model_id} &nbsp;|&nbsp;
   <b>Prompt:</b> "{prompt}" &nbsp;|&nbsp;
   <b>Samples:</b> {len(records)}</p>
<div class="grid">{cards}</div>
</body></html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nFetching {args.n_samples} COCO samples …")
    samples = fetch_samples(args.n_samples)

    # Load model
    model, processor = load_model(args.model_id, args.dtype)

    # Generate
    records = []
    print(f"\nGenerating captions (prompt: \"{args.prompt}\") …")
    for i, s in enumerate(tqdm(samples, desc="Generating")):
        generated = generate_caption(
            model, processor, s["image"],
            args.prompt, args.max_new_tokens, device
        )
        # Save image to disk
        img_path = img_dir / f"sample_{i:03d}_coco{s['coco_id']}.jpg"
        s["image"].save(img_path, "JPEG", quality=90)

        records.append({
            "sample_idx":  i,
            "coco_id":     s["coco_id"],
            "url":         s["url"],
            "img_file":    str(img_path.name),
            "generated":   generated,
            "refs":        s["refs"],
            "image":       s["image"],   # kept only for HTML, not saved to JSON
        })
        print(f"\n[{i:3d}] Generated : {generated}")
        print(f"       Ref[0]    : {s['refs'][0].strip()}")

    # Save captions JSON (without PIL objects)
    caption_data = [{k: v for k, v in r.items() if k != "image"} for r in records]
    json_path = out / "captions.json"
    with open(json_path, "w") as f:
        json.dump({"model_id": args.model_id, "prompt": args.prompt,
                   "samples": caption_data}, f, indent=2)

    # Save HTML report
    html_path = out / "report.html"
    with open(html_path, "w") as f:
        f.write(build_html(records, args.prompt, args.model_id))

    print(f"\n{'='*55}")
    print(f"  Images   : {img_dir}/")
    print(f"  Captions : {json_path}")
    print(f"  Report   : {html_path}   ← open in browser")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
