#!/usr/bin/env python
"""
Capabilities-vs-training-progress sweep.

Evaluates every ckpt_*.pt in a checkpoint directory on the benchmark task set
and plots each metric against training tokens, with the public `gpt2` as a
dashed reference. Shows how downstream capability emerges over pretraining.

Uses --limit by default (we want the trend fast; the final checkpoint's full
numbers live in EVAL.md). Reuses the converter + eval plumbing from
eval_lm_harness.py.

    python benchmarks/checkpoint_sweep.py \
        --checkpoint-dir /mnt/localssd/gpt2/checkpoints --limit 2000
"""
import argparse
import glob
import json
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_lm_harness import (  # noqa: E402
    DEFAULT_TASKS, convert_to_hf, parse_results, run_lm_eval,
)
from gpt2_small import Transformer, TransformerConfig  # noqa: E402

TOKENS_PER_STEP = 512 * 1024  # batch 32 * accum 16 * seq 1024

# Validated categorical palette (dataviz skill), fixed slot order + markers
# as secondary encoding. Perplexity is plotted separately (different axis).
SERIES = [
    ("hellaswag", "acc_norm", "#2a78d6", "o"),
    ("lambada_openai", "acc", "#008300", "s"),
    ("piqa", "acc_norm", "#e87ba4", "^"),
    ("arc_easy", "acc_norm", "#eda100", "D"),
    ("sciq", "acc_norm", "#1baf7a", "v"),
]
PPL_TASK = "wikitext"


def _ckpt_step(path):
    ckpt = t.load(path, map_location="cpu", weights_only=False)
    return int(ckpt["step"]), ckpt


def eval_checkpoint(ckpt, out_dir, tag, tasks, limit, device, batch_size):
    cfg = TransformerConfig(**ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    hf = convert_to_hf(model.eval(), cfg)
    hf_dir = os.path.join(out_dir, f"hf_{tag}")
    hf.save_pretrained(hf_dir)
    # lm-eval needs the tokenizer alongside the model, or it tokenizes to empty.
    from transformers import GPT2TokenizerFast
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(hf_dir)
    res_dir = os.path.join(out_dir, f"res_{tag}")
    run_lm_eval(f"pretrained={hf_dir},dtype=bfloat16", tasks, res_dir, batch_size, limit, device)
    return parse_results(res_dir)


def run_sweep(args):
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ckpt_*.pt")))
    if not paths:
        raise SystemExit(f"No checkpoints in {args.checkpoint_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    points = []  # (step, {task: (metric, val)})
    for path in paths:
        step, ckpt = _ckpt_step(path)
        print(f"\n=== checkpoint step {step} ({os.path.basename(path)}) ===")
        metrics = eval_checkpoint(ckpt, args.out_dir, f"s{step}", args.tasks,
                                  args.limit, args.device, args.batch_size)
        points.append((step, metrics))
    points.sort(key=lambda x: x[0])

    print("\n=== public gpt2 reference ===")
    from eval_lm_harness import run_lm_eval as _r
    gpt2_dir = os.path.join(args.out_dir, "res_gpt2_sweep")
    _r("pretrained=gpt2,dtype=bfloat16", args.tasks, gpt2_dir, args.batch_size, args.limit, args.device)
    gpt2 = parse_results(gpt2_dir)

    data = {
        "tokens_per_step": TOKENS_PER_STEP,
        "limit": args.limit,
        "points": [{"step": s, "tokens": s * TOKENS_PER_STEP, "metrics": m} for s, m in points],
        "gpt2": gpt2,
    }
    json_path = os.path.join(args.out_dir, "sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved -> {json_path}")
    return data


def plot(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    xs = [p["tokens"] / 1e9 for p in data["points"]]

    fig, (ax, axp) = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    for a in (ax, axp):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for s in a.spines.values():
            s.set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=9)

    # --- accuracy tasks ---
    for task, metric, color, marker in SERIES:
        ys = [p["metrics"].get(task, (None, None))[1] for p in data["points"]]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        px, py = zip(*pts)
        ax.plot(px, py, color=color, marker=marker, markersize=6, linewidth=2,
                zorder=5, label=task.replace("_openai", ""))
        ax.annotate(task.replace("_openai", ""), (px[-1], py[-1]),
                    color=color, fontsize=9, fontweight="bold",
                    xytext=(6, 0), textcoords="offset points", va="center")
        gref = data["gpt2"].get(task)
        if gref:
            ax.axhline(gref[1], color=color, linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.55, zorder=3)

    ax.set_xlabel("training tokens (billions)", color=INK2, fontsize=10)
    ax.set_ylabel("accuracy (acc_norm / acc)", color=INK2, fontsize=10)
    ax.set_title("Downstream capability vs. pretraining  (solid = ours, dashed = GPT-2)",
                 color=INK, fontsize=11, fontweight="bold", loc="left")

    # --- perplexity (separate axis; lower is better) ---
    yp = [p["metrics"].get(PPL_TASK, (None, None))[1] for p in data["points"]]
    ppts = [(x, y) for x, y in zip(xs, yp) if y is not None]
    if ppts:
        ppx, ppy = zip(*ppts)
        axp.plot(ppx, ppy, color="#4a3aa7", marker="o", markersize=6, linewidth=2, zorder=5, label="ours")
        gref = data["gpt2"].get(PPL_TASK)
        if gref:
            axp.axhline(gref[1], color="#4a3aa7", linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.6, zorder=3)
            axp.annotate(f"GPT-2 {gref[1]:.1f}", (ppx[-1], gref[1]), color="#4a3aa7",
                         fontsize=9, xytext=(4, 4), textcoords="offset points")
    axp.set_xlabel("training tokens (billions)", color=INK2, fontsize=10)
    axp.set_ylabel("wikitext word perplexity  (lower better)", color=INK2, fontsize=10)
    axp.set_title("Language-modeling quality", color=INK, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("GPT-2 small (124M) on 10B FineWeb-Edu: capability emergence",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", default="/mnt/localssd/gpt2/checkpoints")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", default="/mnt/localssd/gpt2/sweep")
    ap.add_argument("--limit", type=int, default=2000, help="examples/task (trend, not final numbers)")
    ap.add_argument("--batch-size", default="64")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plot-only", action="store_true", help="replot from saved sweep_results.json")
    args = ap.parse_args()

    if args.plot_only:
        data = json.load(open(os.path.join(args.out_dir, "sweep_results.json")))
    else:
        data = run_sweep(args)
    plot(data, os.path.join(args.out_dir, "capability_emergence.png"))


if __name__ == "__main__":
    main()
