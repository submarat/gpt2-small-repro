#!/usr/bin/env python
"""
Throughput / memory benchmark for the gpt2_small Transformer.

Isolates *model compute* (no dataset download) using synthetic random token
batches so before/after optimization numbers are clean and reproducible.

Measures, at the full GPT-2 small config by default:
  * train step: forward + next-token cross-entropy loss + backward + optimizer step
  * eval step : forward only (torch.no_grad), i.e. what generation/eval pays

For each it reports median step time, tokens/sec, and peak GPU memory.

Optimizations are exposed as flags so the SAME harness measures baseline vs.
optimized:
    baseline : --dtype fp32
    optimized: --dtype bf16 --compile
The SDPA-attention change lives in gpt2_small/model.py and is picked up
automatically (no flag needed).

Examples
--------
    # baseline
    python benchmarks/benchmark.py --tag baseline --dtype fp32
    # after enabling bf16 + torch.compile
    python benchmarks/benchmark.py --tag optimized --dtype bf16 --compile
    # compare two saved runs
    python benchmarks/benchmark.py --compare baseline optimized
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch as t

from gpt2_small import Transformer, TransformerConfig, device

RESULTS_DIR = Path(__file__).parent / "results"


def next_token_loss(logits: t.Tensor, tokens: t.Tensor, xent: bool = False) -> t.Tensor:
    """Mean next-token cross-entropy loss.

    xent=False mirrors gpt2_small.training.train's manual log_softmax+gather;
    xent=True uses the fused F.cross_entropy.
    """
    if xent:
        return t.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(
        dim=-1, index=tokens[:, 1:].unsqueeze(-1)
    )
    return -log_probs_for_tokens.mean()


def _sync() -> None:
    if device.type == "cuda":
        t.cuda.synchronize()


def _autocast(dtype: str):
    """Return an autocast context for bf16/fp16, or a no-op for fp32."""
    if dtype == "fp32" or device.type != "cuda":
        return t.autocast(device_type="cpu", enabled=False)
    torch_dtype = t.bfloat16 if dtype == "bf16" else t.float16
    return t.autocast(device_type="cuda", dtype=torch_dtype)


def _time_loop(step_fn, warmup: int, iters: int) -> list[float]:
    """Run warmup (untimed) then `iters` timed steps; return per-step seconds."""
    for _ in range(warmup):
        step_fn()
    _sync()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        step_fn()
        _sync()
        times.append(time.perf_counter() - start)
    return times


def _summarize(name: str, times: list[float], n_tokens: int) -> dict:
    median = statistics.median(times)
    peak_gb = (
        t.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    )
    return {
        "phase": name,
        "step_ms_median": median * 1e3,
        "step_ms_min": min(times) * 1e3,
        "tokens_per_sec": n_tokens / median,
        "peak_mem_gb": peak_gb,
    }


def run(args) -> dict:
    t.manual_seed(0)
    if args.tf32 and device.type == "cuda":
        t.set_float32_matmul_precision("high")
    cfg = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_head=args.d_model // args.n_heads,
        d_mlp=4 * args.d_model,
        seq_len=args.seq_len,
        vocab=args.vocab,
        use_sdpa=args.sdpa,
        tie_weights=args.tie,
    )
    model = Transformer(cfg).to(device)
    model.config = cfg
    n_params = sum(p.numel() for p in model.parameters())

    if args.compile:
        model = t.compile(model)

    tokens = t.randint(0, cfg.vocab, (args.batch, cfg.seq_len), device=device)
    n_tok = args.batch * cfg.seq_len
    optim = t.optim.AdamW(model.parameters(), lr=1e-4, fused=args.fused_adam)

    def train_step():
        optim.zero_grad(set_to_none=True)
        with _autocast(args.dtype):
            logits = model(tokens)
            loss = next_token_loss(logits, tokens, xent=args.xent)
        loss.backward()
        optim.step()

    def eval_step():
        with t.no_grad(), _autocast(args.dtype):
            model(tokens)

    results = {
        "tag": args.tag,
        "device": device.type,
        "gpu": t.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "compile": args.compile,
        "tf32": args.tf32,
        "sdpa": args.sdpa,
        "fused_adam": args.fused_adam,
        "xent": args.xent,
        "tie": args.tie,
        "config": {
            "d_model": cfg.d_model, "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers, "d_mlp": cfg.d_mlp,
            "seq_len": cfg.seq_len, "vocab": cfg.vocab,
        },
        "batch": args.batch,
        "n_params": n_params,
        "warmup": args.warmup,
        "iters": args.iters,
        "phases": {},
    }

    # Train
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats()
    train_times = _time_loop(train_step, args.warmup, args.iters)
    results["phases"]["train"] = _summarize("train", train_times, n_tok)

    # Eval (forward only)
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats()
    eval_times = _time_loop(eval_step, args.warmup, args.iters)
    results["phases"]["eval"] = _summarize("eval", eval_times, n_tok)

    return results


def print_results(r: dict) -> None:
    print("=" * 68)
    print(f"  tag={r['tag']}  dtype={r['dtype']}  compile={r['compile']}")
    print(f"  {r['gpu']}  |  params={r['n_params']:,}")
    c = r["config"]
    print(
        f"  d_model={c['d_model']} n_layers={c['n_layers']} n_heads={c['n_heads']} "
        f"seq_len={c['seq_len']} batch={r['batch']}"
    )
    print("-" * 68)
    print(f"  {'phase':<7}{'step (ms)':>12}{'tokens/sec':>16}{'peak mem (GB)':>16}")
    for name, p in r["phases"].items():
        print(
            f"  {name:<7}{p['step_ms_median']:>12.2f}"
            f"{p['tokens_per_sec']:>16,.0f}{p['peak_mem_gb']:>16.2f}"
        )
    print("=" * 68)


def compare(tag_a: str, tag_b: str) -> None:
    def load(tag):
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            raise SystemExit(f"No saved results for tag '{tag}' at {path}")
        return json.loads(path.read_text())

    a, b = load(tag_a), load(tag_b)
    print(f"\nComparison: {tag_a} -> {tag_b}\n" + "=" * 68)
    hdr = f"  {'phase':<7}{'metric':<16}{tag_a:>14}{tag_b:>14}{'speedup':>10}"
    print(hdr + "\n" + "-" * 68)
    for phase in a["phases"]:
        pa, pb = a["phases"][phase], b["phases"][phase]
        tps_a, tps_b = pa["tokens_per_sec"], pb["tokens_per_sec"]
        print(
            f"  {phase:<7}{'tokens/sec':<16}{tps_a:>14,.0f}{tps_b:>14,.0f}"
            f"{tps_b / tps_a:>9.2f}x"
        )
        print(
            f"  {'':<7}{'peak mem (GB)':<16}{pa['peak_mem_gb']:>14.2f}"
            f"{pb['peak_mem_gb']:>14.2f}{pb['peak_mem_gb'] / pa['peak_mem_gb']:>9.2f}x"
        )
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="run", help="label; results saved to results/<tag>.json")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--compile", action="store_true", help="wrap model in torch.compile")
    ap.add_argument("--tf32", action="store_true", help="set_float32_matmul_precision('high')")
    ap.add_argument("--sdpa", action="store_true", help="use fused scaled_dot_product_attention")
    ap.add_argument("--fused-adam", action="store_true", help="AdamW(fused=True)")
    ap.add_argument("--xent", action="store_true", help="use F.cross_entropy loss")
    ap.add_argument("--tie", action="store_true", help="tie unembedding to embedding weight")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="print comparison of two saved runs and exit")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if device.type != "cuda":
        print("WARNING: CUDA not available; benchmarking on CPU is not meaningful.")

    results = run(args)
    print_results(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
