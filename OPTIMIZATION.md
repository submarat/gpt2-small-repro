# Optimization Log

Tracking throughput/memory optimizations for the `gpt2_small` Transformer, with
the goal of making a full GPT-2-small run on OpenWebText / FineWeb practical on a
single H100.

## Method

All numbers come from `benchmarks/benchmark.py`, which uses synthetic random-token
batches (no dataset download) to isolate model compute. It reports median step
time, tokens/sec, and peak GPU memory for two phases:

- **train** — forward + next-token cross-entropy loss + backward + optimizer step
- **eval** — forward only (`torch.no_grad`)

Run baseline vs. optimized with the same harness (optimizations are flags; the
SDPA-attention change lives in `model.py`):

```bash
python benchmarks/benchmark.py --tag baseline  --dtype fp32
python benchmarks/benchmark.py --tag optimized --dtype bf16 --compile
python benchmarks/benchmark.py --compare baseline optimized
```

## Environment

| | |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3 (GPU 0; GPU 1 in use by another process) |
| CPU / RAM | 192× AMD EPYC 7R13, 2.0 TB RAM |
| torch | 2.13.0+cu130 |
| Python | 3.11 |
| venv | `/mnt/localssd/gpt2/venv` (root disk is near-full; keep everything on `/mnt/localssd`) |

## Benchmark config

Full GPT-2 small: `d_model=768, n_layers=12, n_heads=12, d_head=64, d_mlp=3072,
vocab=50257, seq_len=1024`, `batch=8`, 5 warmup + 20 timed steps.

Note: this implementation does **not** tie embedding/unembedding weights, so it
reports **163M params** vs. GPT-2's 124M (separate 50257×768 `W_E` and `W_U`).

## Results

Full GPT-2 small, batch 8, seq 1024, on one H100 80GB. Each step is cumulative
(includes all previous). "x" columns are speedup vs. the **previous** step.

### Train (fwd + loss + bwd + optim)

| # | tag | step (ms) | tokens/sec | x prev | peak mem (GB) |
|---|---|---|---|---|---|
| 0 | baseline   | 223.83 | 36,599  | —     | 19.69 |
| 1 | tf32       | 122.93 | 66,639  | 1.82x | 19.69 |
| 2 | bf16       | 116.90 | 70,079  | 1.05x | 21.07 |
| 3 | sdpa       | 76.52  | 107,060 | 1.53x | 13.83 |
| 4 | compile    | 49.97  | 163,923 | 1.53x | 11.69 |
| 5 | fused_adam | 44.32  | 184,839 | 1.13x | 11.69 |
| 6 | xent       | 45.81  | 178,832 | 0.97x | 11.68 |
| 7 | tied       | 45.88  | 178,543 | 1.00x | 11.21 |

### Eval (forward only)

| # | tag | step (ms) | tokens/sec | x prev | peak mem (GB) |
|---|---|---|---|---|---|
| 0 | baseline   | 75.67 | 108,262 | —     | 6.05 |
| 1 | tf32       | 38.71 | 211,605 | 1.95x | 6.05 |
| 2 | bf16       | 39.08 | 209,640 | 0.99x | 5.52 |
| 3 | sdpa       | 22.37 | 366,282 | 1.75x | 5.54 |
| 4 | compile    | 10.89 | 751,976 | 2.05x | 5.26 |
| 5 | fused_adam | 10.91 | 750,810 | 1.00x | 5.20 |
| 6 | xent       | 10.99 | 745,422 | 0.99x | 5.20 |
| 7 | tied       | 12.77 | 641,744 | 0.86x | 4.59 |

**Params:** 163,087,441 (steps 0–6) → **124,490,065** after tying (step 7),
matching GPT-2 small.

### Bottom line (baseline → fully optimized)

| metric | baseline | optimized | change |
|---|---|---|---|
| train tokens/sec | 36,599 | 178,543 | **4.88x** |
| eval tokens/sec  | 108,262 | 641,744 | **5.93x** |
| train peak mem   | 19.69 GB | 11.21 GB | −43% |
| params           | 163.1M | 124.5M | −38.6M (GPT-2 parity) |

### Observations

- **tf32 was the single biggest free win** (~1.8–1.95x): the baseline was doing
  fp32 matmuls on non-tensor-core paths. One line, no accuracy concern for training.
- **bf16 on top of tf32 was small** (~1.05x / neutral) — tf32 already used tensor
  cores. Its main value is enabling the flash SDPA kernel and lower activation mem.
- **sdpa** and **compile** were the two big structural wins (~1.5–2x each) and cut
  train memory from 21 → 11.7 GB by not materializing the `[b, h, seq, seq]` probs.
- **fused_adam** gave a real train-only ~1.13x (no eval effect, as expected).
- **xent** was throughput-neutral (within noise); kept for numerical stability and
  because it removes the redundant intermediate tensors.
- **tied**: as predicted, a *storage* win, not FLOPs — params 163M → 124.5M, train
  mem 11.69 → 11.21 GB, train throughput flat. Eval regressed ~14%: the tied path
  multiplies by the transposed `(vocab, d_model)` embedding, whose einsum orientation
  compiles to a less efficient matmul than the native `(d_model, vocab)` `W_U`.
  Acceptable for training (train throughput unaffected); revisit if eval/generation
  latency matters (e.g. materialize a contiguous transpose once).

## Optimization steps

Applied **cumulatively** and measured **one at a time** so each step's gain is
attributable. Each step gets its own benchmark tag; compare against the previous
tag with `benchmark.py --compare <prev> <this>`. Every step must keep
`tests/test_model.py` passing.

| # | tag | change | where | expected effect | status |
|---|---|---|---|---|---|
| 0 | `baseline` | fp32, no compile | — | reference | ✅ done |
| 1 | `tf32` | `set_float32_matmul_precision("high")` (TF32 matmul/cuDNN) | train entrypoint | faster fp32 matmuls, ~free | ✅ |
| 2 | `bf16` | bf16 autocast for fwd/bwd | train loop + benchmark flag | large speedup + lower mem on H100 tensor cores | ✅ |
| 3 | `sdpa` | `F.scaled_dot_product_attention` (fused/flash) replaces einsum+softmax | `model.py` `MultiHeadAttention` | large speedup + big mem drop; probs-return becomes optional | ✅ |
| 4 | `compile` | `torch.compile(model)` | train loop + benchmark flag | kernel fusion speedup | ✅ |
| 5 | `fused_adam` | `AdamW(..., fused=True)` | train loop | faster optimizer step | ✅ |
| 6 | `xent` | replace manual `log_softmax`+`gather` with `F.cross_entropy` | `training.py` + benchmark loss | fused, stabler, slightly faster train step | ✅ |
| 7 | `tied` | tie `W_U = W_E.T` (share one table) | `model.py` Embedding/Unembedding | −38M params (163M→~124M), matches GPT-2, lower mem | ✅ |

## End-to-end training throughput (fully optimized)

Batch sweep of the fully-optimized model (`benchmark.py`, bf16+sdpa+compile+
fused_adam+xent+tie). Throughput plateaus ~200K tok/s (compute-bound); memory
scales ~linearly. **batch 32 (~39 GB)** is the sweet spot on 80 GB.

| micro-batch | train tok/s | peak mem (GB) |
|---|---|---|
| 8  | 178,543 | 11.21 |
| 16 | 192,985 | 20.48 |
| 32 | 201,041 | 39.11 |
| 48 | 205,115 | 57.74 |

Real end-to-end run on WikiText-2 (`benchmarks/train_throughput.py`), full
`train()` path incl. gradient accumulation, batch 32 × accum 16 = **524,288
tokens/opt-step** (GPT-2's batch), measured over 10 steps after warmup:

| metric | value |
|---|---|
| tokens/sec | **202,295** (matches synthetic → dataloader overhead negligible) |
| sec / opt-step (524K tok) | 2.59 |
| peak mem | 39.6 GB |
| params | 124.5M (tied) |

**Projected single-H100 wall-clock:** ~12.4 h per 9B-token epoch (OWT);
~17 days for a 300B-token GPT-2-grade run.

### Training-loop items (correctness/scale, not micro-benchmarked here)

These affect a real OWT/FineWeb run but not the fixed-batch micro-benchmark, so
they're tracked separately and validated on a short real training run:

- [x] **Gradient accumulation** — `train(grad_accum_steps=N)`; effective batch =
      `batch_size * N` seqs. Loss scaled by 1/N; verified gradient-equivalent to a
      single full batch (max diff ~4e-9).
- [x] **DataLoader** — fixed loaders for newer datasets/huggingface_hub
      (canonical repo ids + lazy Arrow collate), `num_workers`/`pin_memory`
      enabled; added OpenWebText and FineWeb-Edu loaders. Spurious trailing
      `.unsqueeze(-1)` removed by switching to `F.cross_entropy`.
- [x] **LR schedule + gradient clipping** — cosine decay with linear warmup
      (`cosine_lr`), `clip_grad_norm_(1.0)`, AdamW (0.9, 0.95) with the >=2D-only
      weight-decay split (`configure_optimizer`).
- [x] **Step-driven pretraining loop** — token-budget (`max_steps`) stopping,
      periodic validation (`estimate_loss`), checkpointing (`save_checkpoint`),
      and W&B logging of loss/lr/grad_norm/tokens_per_sec. Entry point:
      `demos/train_gpt2_fineweb.py` (10B FineWeb-Edu by default).

## How each step is recorded

After applying step _N_:

```bash
python benchmarks/benchmark.py --tag <tag>            # + relevant flags
python benchmarks/benchmark.py --compare <prev> <tag> # attribute the gain
```

Then append a row to **Results** and tick the box above.
