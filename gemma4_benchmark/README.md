# Gemma 4 E4B — Image-to-Text Baseline Benchmark

Baseline evaluation of [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) on English image captioning before compression. The goal of this project is to compress the model while retaining **≥ 80% of baseline quality** and minimising energy consumption.

---

## Setup

**Hardware used:** GCP VM — NVIDIA L4 (22 GiB VRAM), Intel Xeon @ 2.20 GHz, 15 GiB RAM  
**Software:** Python 3.11 · PyTorch 2.11+cu128 · Transformers 5.6.2 · CodeCarbon 3.2.6

```bash
pip install -r requirements.txt
```

> The NVIDIA driver (595) and CUDA toolkit (12.8) must be installed.  
> On a fresh GCP Debian 12 VM follow the driver setup in the [driver notes](#driver-setup) section below.

---

## Benchmark

Runs inference on COCO Karpathy test images, measures quality metrics (BLEU-4, ROUGE-L, METEOR) and energy consumption via [CodeCarbon](https://github.com/mlco2/codecarbon).

```bash
# Full 100-sample baseline
python3 benchmark.py --n_samples 100 --dtype bfloat16

# Quick smoke test
python3 benchmark.py --n_samples 10
```

Results are saved to `results/baseline_<timestamp>.json`.

---

## Baseline Results — Run `20260428_152538`

**Model:** `google/gemma-4-E4B` (7.94 B params, bfloat16)  
**Dataset:** COCO Karpathy test split (`yerevann/coco-karpathy`) — 100 samples  
**Prompt:** `"Describe this image in one sentence."`

### Quality

| Metric | Score | 80% Compression Target |
|--------|-------|------------------------|
| BLEU-4 | 0.0305 | ≥ 0.0244 |
| ROUGE-L | 0.1683 | ≥ 0.1346 |
| METEOR | 0.1932 | ≥ 0.1546 |

> **Note on low BLEU-4:** `gemma-4-E4B` is a **base model** (not instruction-tuned). It generates blog-style continuations — including keywords, tags, and repetition loops — rather than clean one-sentence captions. BLEU-4 is extremely sensitive to n-gram mismatch, so even semantically correct outputs score low. METEOR (0.1932) is the more meaningful metric here as it accounts for stemming and synonyms. See [Output Inspection](#output-inspection) below.

### Speed

| Metric | Value |
|--------|-------|
| Total time (100 samples) | 653 s (10.9 min) |
| Mean latency per sample | 6,533 ms |
| Std dev latency | 4,731 ms |
| Throughput | 0.153 samples / sec |

The high latency variance (σ = 4,731 ms) is driven by dynamic image patch counts — larger images produce more vision tokens and longer forward passes.

### Energy & CO₂

| Metric | Total | Per Sample |
|--------|-------|------------|
| Energy | 20.85 Wh | 0.208 Wh |
| CO₂ | 8.96 gCO₂ | 89.6 µgCO₂ |

**Breakdown (CodeCarbon):**

| Component | Energy (Wh) | Share | Avg Power |
|-----------|-------------|-------|-----------|
| GPU (NVIDIA L4) | 11.32 | 54% | 62.4 W / 72 W max |
| CPU (Xeon) | 7.71 | 37% | 42.5 W (TDP estimate) |
| RAM | 1.81 | 9% | — |

GPU utilisation averaged **55%** — not saturated because batch size is 1. After compression, increasing the batch size should significantly improve throughput per Wh.

### Memory

| Metric | Value |
|--------|-------|
| Peak VRAM | 15,675 MiB (15.3 GiB) |
| Total VRAM | 22,565 MiB (22.0 GiB) |
| Headroom | ~6.7 GiB free |

---

## Output Inspection

```bash
python3 inspect_outputs.py --n_samples 20
```

Saves to `inspect_outputs/`:
- `images/` — raw COCO images  
- `captions.json` — generated + reference captions per sample  
- `report.html` — browser-viewable side-by-side comparison (images embedded as base64)

### Sample outputs (20 inspected)

| # | Generated | Reference[0] | Observation |
|---|-----------|--------------|-------------|
| 0 | *"This image is about a man riding a motorcycle on a road. Keywords: Road, motorcycle…"* | *"A man with a red helmet on a small moped on a dirt road."* | Correct content, blog-style suffix hurts BLEU |
| 1 | *"The girl is eating a bowl of soup."* | *"A young girl inhales with the intent of blowing out a candle."* | Plausible but wrong action |
| 2 | *"A man riding a bicycle on a road next to a train track. Where did you capture…"* | *"A man on a bicycle riding next to a train"* | Core caption correct, model continues as Q&A |
| 5 | *"The image shows a table with a collection of wooden spoons and other kitchen utensils."* | *"Multiple wooden spoons are shown on a table top."* | Near-perfect semantically |
| 11 | *"A young man in a black shirt and tie smiles at the camera. Keywords…"* | *"A young man wearing black attire and a flowered tie is standing and smiling."* | Correct, keyword suffix degrades BLEU |
| 17 | *"This is a picture of a motorcycle. This is a picture of a motorcycle. …"* | *"A motorbike sitting in front of a wine display case"* | Repetition loop — most common base-model failure |

**Root causes of low scores:**

1. **Blog-style generation** — the base model continues text as if writing an article (adds Keywords, Tags, Q&A sections). These extra tokens destroy n-gram matching.
2. **Repetition loops** — without instruction tuning, greedy decoding can fall into repetition (e.g. sample 17).
3. **Semantic correctness is reasonable** — the first sentence is usually on-topic. Extracting only the first sentence would substantially raise ROUGE-L and METEOR.

**Implication for compression:** since the baseline and compressed model use the same pipeline and prompt, relative scores (compressed / baseline) are still meaningful. The 80% threshold is applied to these relative values.

---

## File Structure

```
gemma4_benchmark/
├── benchmark.py          # Main benchmark — quality metrics + energy tracking
├── inspect_outputs.py    # Visual inspection — saves images, captions, HTML report
├── requirements.txt      # Python dependencies
├── results/
│   ├── baseline_20260428_152538.json   # Full baseline metrics
│   └── emissions.csv                   # Raw CodeCarbon output
└── inspect_outputs/
    ├── images/           # 20 COCO test images
    ├── captions.json     # Generated + reference captions
    └── report.html       # Side-by-side visual report
```

---

## Driver Setup

On a fresh GCP Debian 12 VM with an NVIDIA L4:

```bash
# Add NVIDIA CUDA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install driver + kernel headers (builds DKMS module)
sudo apt-get install -y cuda-drivers linux-headers-$(uname -r)

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-8

# Load kernel modules
sudo modprobe nvidia && sudo modprobe nvidia-uvm

# Verify
nvidia-smi
```

---

## Next Steps

- [ ] Layer-similarity analysis to identify redundant blocks (using `Short-LLM-main/`)
- [ ] Structural pruning — remove identified blocks
- [ ] Re-run `benchmark.py` on pruned model, compare against 80% thresholds
- [ ] Fine-tune pruned model if quality drops below threshold
- [ ] Energy comparison: baseline vs compressed (Wh/sample)
