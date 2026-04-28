"""
Gemma 4 E4B — image-to-text baseline benchmark.

Measures:
  - Caption quality  : BLEU-4, ROUGE-L, METEOR
  - Inference speed  : samples/sec, ms/sample
  - Energy / CO₂     : kWh and gCO₂eq via CodeCarbon
  - GPU memory        : peak VRAM (MiB)

Dataset  : yerevann/coco-karpathy  (Parquet, images fetched via COCO public URL)
Model ID : google/gemma-4-e4b-it  (override via --model_id)

Results are saved to results/baseline_<timestamp>.json for later
comparison after compression.
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from io import BytesIO

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import evaluate
import nltk
from codecarbon import EmissionsTracker


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gemma 4 E4B image-to-text benchmark")
    p.add_argument("--model_id",    default="google/gemma-4-E4B",
                   help="HuggingFace model ID")
    p.add_argument("--n_samples",   type=int, default=100,
                   help="Number of images to evaluate (default 100)")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max tokens to generate per image")
    p.add_argument("--batch_size",  type=int, default=1,
                   help="Inference batch size (keep 1 for generative models)")
    p.add_argument("--prompt",      default="Describe this image in one sentence.",
                   help="Text prompt sent alongside each image")
    p.add_argument("--dataset",
                   default="yerevann/coco-karpathy",
                   help="HuggingFace image-captioning dataset")
    p.add_argument("--results_dir", default="results",
                   help="Directory to write JSON results")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype")
    return p.parse_args()


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def load_coco_karpathy(n_samples: int) -> list:
    """
    yerevann/coco-karpathy — COCO 2014 Karpathy test split (Parquet, no script).
    Images are fetched from the `url` column (COCO public server).
    `sentences` is a list of 5 English reference caption strings per image.
    """
    ds = load_dataset("yerevann/coco-karpathy", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    samples = []
    session = requests.Session()
    for item in tqdm(ds, desc="Fetching COCO images", leave=False):
        try:
            resp = session.get(item["url"], timeout=15)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"\n  Warning: could not fetch {item['url']}: {e} — skipping")
            continue
        captions = item["sentences"]
        if not isinstance(captions, list):
            captions = [captions]
        samples.append((image, captions))
    return samples


# ── Model helpers ────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def load_model(model_id: str, dtype_str: str):
    dtype = DTYPE_MAP[dtype_str]
    print(f"\nLoading processor from {model_id} …")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model ({dtype_str}) …")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, processor


def build_messages(prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.inference_mode()
def generate_caption(model, processor, image: Image.Image, prompt: str,
                     max_new_tokens: int, device: str) -> str:
    # Try chat-template format first (works for -it models).
    # Base models (e.g. google/gemma-4-E4B) have no chat template.
    # For those, the text must contain exactly one processor.image_token
    # placeholder per image — the processor then expands it to
    # <boi><|image|>*n<eoi> where n = actual patch count.
    try:
        messages = build_messages(prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        image_token = getattr(processor, "image_token", "<|image|>")
        text = f"{image_token}\n{prompt}"

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[str], references: list[list[str]]) -> dict:
    """
    predictions : one string per sample
    references  : list of reference caption lists per sample
    Returns dict with bleu4, rougeL, meteor scores.
    """
    # BLEU-4 (corpus level)
    nltk.download("punkt",        quiet=True)
    nltk.download("punkt_tab",    quiet=True)
    nltk.download("wordnet",      quiet=True)
    nltk.download("omw-1.4",      quiet=True)

    bleu_metric   = evaluate.load("bleu")
    rouge_metric  = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # evaluate library expects refs as list-of-lists for BLEU
    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=references,   # list[list[str]]
    )

    # ROUGE and METEOR expect a single reference string → use first ref
    single_refs = [refs[0] for refs in references]
    rouge_result  = rouge_metric.compute(predictions=predictions, references=single_refs)
    meteor_result = meteor_metric.compute(predictions=predictions, references=single_refs)

    return {
        "bleu4":   round(bleu_result.get("bleu", 0.0), 4),
        "rougeL":  round(rouge_result.get("rougeL", 0.0), 4),
        "meteor":  round(meteor_result.get("meteor", 0.0), 4),
    }


# ── GPU memory helper ────────────────────────────────────────────────────────

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset '{args.dataset}' ({args.n_samples} samples) …")
    samples = load_coco_karpathy(args.n_samples)

    print(f"Loaded {len(samples)} samples.")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model_id, args.dtype)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e9:.2f}B")

    # ── Run inference with energy tracking ────────────────────────────────────
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="gemma4_e4b_baseline",
        output_dir=args.results_dir,
        log_level="warning",
        save_to_file=True,
    )

    predictions: list[str] = []
    references:  list[list[str]] = []
    inference_times: list[float] = []

    reset_peak_memory()
    tracker.start()
    t_total_start = time.perf_counter()

    for image, captions in tqdm(samples, desc="Generating captions"):
        t0 = time.perf_counter()
        caption = generate_caption(
            model, processor, image,
            args.prompt, args.max_new_tokens, device,
        )
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)  # ms
        predictions.append(caption)
        references.append(captions)

    t_total_end = time.perf_counter()
    emissions = tracker.stop()  # gCO₂eq (codecarbon ≥2.x returns kg → multiply)

    total_time_s = t_total_end - t_total_start
    mean_ms      = float(np.mean(inference_times))
    std_ms       = float(np.std(inference_times))
    throughput   = len(samples) / total_time_s

    # ── Compute quality metrics ───────────────────────────────────────────────
    print("\nComputing quality metrics …")
    quality = compute_metrics(predictions, references)

    # ── Read energy from CodeCarbon CSV (more reliable than return value) ─────
    cc_csv = Path(args.results_dir) / "emissions.csv"
    energy_kwh = 0.0
    co2_kg     = 0.0
    if cc_csv.exists():
        import csv
        with open(cc_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            energy_kwh = float(last.get("energy_consumed", 0.0))
            co2_kg     = float(last.get("emissions",       0.0))

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "run_id":    datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "model_id":  args.model_id,
        "dtype":     args.dtype,
        "n_params_B": round(n_params / 1e9, 3),
        "dataset":   args.dataset,
        "n_samples": len(samples),
        "prompt":    args.prompt,
        "quality": {
            **quality,
            "description": (
                "bleu4: corpus BLEU-4 | "
                "rougeL: longest common subsequence F1 | "
                "meteor: unigram F1 with stemming and synonym matching"
            ),
        },
        "speed": {
            "total_time_s":       round(total_time_s, 2),
            "throughput_samples_per_s": round(throughput, 3),
            "mean_latency_ms":    round(mean_ms, 1),
            "std_latency_ms":     round(std_ms, 1),
        },
        "energy": {
            "energy_kwh":  round(energy_kwh, 6),
            "co2_kg":      round(co2_kg, 6),
            "note": "Measured by CodeCarbon over the full inference loop",
        },
        "hardware": {
            "device":           device,
            "gpu_name":         torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
            "peak_vram_mib":    round(peak_memory_mib(), 1),
            "total_vram_mib":   round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
            ) if device == "cuda" else 0,
        },
    }

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.results_dir) / f"baseline_{results['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Model        : {args.model_id}")
    print(f"  Samples      : {len(samples)}")
    print(f"  BLEU-4       : {quality['bleu4']:.4f}")
    print(f"  ROUGE-L      : {quality['rougeL']:.4f}")
    print(f"  METEOR       : {quality['meteor']:.4f}")
    print(f"  Throughput   : {throughput:.2f} samples/s")
    print(f"  Latency (ms) : {mean_ms:.1f} ± {std_ms:.1f}")
    print(f"  Energy       : {energy_kwh*1000:.3f} Wh  ({co2_kg*1000:.2f} gCO₂)")
    print(f"  Peak VRAM    : {peak_memory_mib():.0f} MiB")
    print(f"  Results saved: {out_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
