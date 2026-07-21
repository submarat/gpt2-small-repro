#!/usr/bin/env python
"""
Mechanistic capstone: watch induction heads form across training.

For every ckpt_*.pt this computes two quantities and plots them vs. training
tokens (same x-axis as the capability sweep), plus a per-head induction-score
heatmap for the final checkpoint:

  * per-head **induction score** - on a sequence of *random* tokens repeated
    twice ([r_1..r_L][r_1..r_L]), an induction head at position p in the second
    copy attends back to position p-L+1 (the token that followed the previous
    occurrence of the current token, i.e. the correct next token). The score is
    the attention mass on that offset, averaged over positions/heads. Random
    tokens remove any bigram-prior confound - the only way to score high is the
    induction mechanism itself.
  * **in-context-learning (ICL) score** = loss@pos50 - loss@pos500 on real text.
    The behavioral signature induction enables: predicting better with more
    context.

The classic result (Olsson et al. 2022) is a phase change - induction score
jumps abruptly during training and the ICL score jumps with it ("induction
bump"). With our 10 checkpoints we can see whether that happens here.

We read attention patterns directly from our own model (use_sdpa=False makes
MultiHeadAttention return probs) via forward hooks - exact weights, exact
(erf) GELU, no conversion.

    python benchmarks/induction_sweep.py --checkpoint-dir /mnt/localssd/gpt2/checkpoints
"""
import argparse
import glob
import json
import os

import torch as t

from gpt2_small import Transformer, TransformerConfig, device
from gpt2_small.training import load_wikitext_dataset

TOKENS_PER_STEP = 512 * 1024


def load_model(ckpt_path):
    ckpt = t.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["config"])
    cfg.use_sdpa = False  # need attention probabilities
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, int(ckpt["step"])


def capture_attention(model, tokens):
    """Run a forward pass, returning {layer: attn_probs [B, H, S, S]}."""
    patterns = {}
    handles = []
    for i, block in enumerate(model.blocks):
        def hook(_mod, _inp, out, i=i):
            patterns[i] = out[0].detach()  # attn module returns (probs, out)
        handles.append(block.attn.register_forward_hook(hook))
    with t.no_grad():
        model(tokens)
    for h in handles:
        h.remove()
    return patterns


@t.no_grad()
def induction_scores(model, cfg, seq_len=128, batch=8, seed=0):
    """Per-head induction score matrix [n_layers, n_heads] in [0, 1]."""
    t.manual_seed(seed)
    rand = t.randint(0, cfg.vocab, (batch, seq_len), device=device)
    tokens = t.cat([rand, rand], dim=1)  # [B, 2*seq_len]
    patterns = capture_attention(model, tokens)
    dest = t.arange(seq_len, 2 * seq_len, device=device)  # second copy
    src = dest - seq_len + 1                               # induction offset
    scores = t.zeros(cfg.n_layers, cfg.n_heads)
    for layer, patt in patterns.items():
        vals = patt[:, :, dest, src]           # [B, H, len(dest)]
        scores[layer] = vals.mean(dim=(0, 2)).cpu()
    return scores


@t.no_grad()
def icl_score(model, tokens_batch, early=50, late=500):
    """loss@early - loss@late on real text (positive = in-context learning)."""
    logits = model(tokens_batch)
    logp = logits.log_softmax(dim=-1)
    tgt = tokens_batch[:, 1:]
    lp = logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
    loss_per_pos = -lp.mean(0)                                   # [S-1]
    return (loss_per_pos[early] - loss_per_pos[late]).item()


def run(args):
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ckpt_*.pt")))
    if not paths:
        raise SystemExit(f"No checkpoints in {args.checkpoint_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    # A fixed batch of real text for the ICL score (seq len 512).
    cfg0 = TransformerConfig()
    text = next(iter(load_wikitext_dataset(cfg0, batch_size=16)))[0][:, :512].to(device)

    rows, final_heatmap = [], None
    for path in paths:
        model, cfg, step = load_model(path)
        heads = induction_scores(model, cfg)
        icl = icl_score(model, text)
        top = heads.flatten().topk(3).values
        rows.append({
            "step": step, "tokens": step * TOKENS_PER_STEP,
            "induction_max": heads.max().item(),
            "induction_top3": top.mean().item(),
            "icl": icl,
        })
        argmax = heads.argmax().item()
        print(f"step {step:>6}: induction_max {heads.max().item():.3f} "
              f"(L{argmax // cfg.n_heads}H{argmax % cfg.n_heads}) | ICL {icl:.3f}")
        final_heatmap = heads.tolist()
        del model
        t.cuda.empty_cache() if device.type == "cuda" else None

    rows.sort(key=lambda r: r["step"])
    data = {"points": rows, "final_heatmap": final_heatmap}
    with open(os.path.join(args.out_dir, "induction_results.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data


def plot(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    BLUE, GREEN = "#2a78d6", "#008300"
    blue_ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", blue_ramp)

    xs = [p["tokens"] / 1e9 for p in data["points"]]
    ind = [p["induction_max"] for p in data["points"]]
    ind3 = [p["induction_top3"] for p in data["points"]]
    icl = [p["icl"] for p in data["points"]]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for s in a.spines.values():
            s.set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=9)
        a.set_xlabel("training tokens (billions)", color=INK2, fontsize=10)

    a1.plot(xs, ind, color=BLUE, marker="o", markersize=6, linewidth=2, label="max head")
    a1.plot(xs, ind3, color=BLUE, marker="o", markersize=4, linewidth=1.3, alpha=0.5, label="mean top-3")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2)
    a1.set_ylabel("induction score (attention on p−L+1)", color=INK2, fontsize=10)
    a1.set_title("Induction head strength vs. training", color=INK, fontsize=11, fontweight="bold", loc="left")

    a2.plot(xs, icl, color=GREEN, marker="s", markersize=6, linewidth=2)
    a2.set_ylabel("ICL score  (loss@50 − loss@500, nats)", color=INK2, fontsize=10)
    a2.set_title("In-context learning vs. training", color=INK, fontsize=11, fontweight="bold", loc="left")

    heat = data["final_heatmap"]
    im = a3.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    a3.set_xlabel("head", color=INK2, fontsize=10)
    a3.set_ylabel("layer", color=INK2, fontsize=10)
    a3.set_title("Per-head induction score (final ckpt)", color=INK, fontsize=11, fontweight="bold", loc="left")
    a3.tick_params(colors=MUTED, labelsize=8)
    cb = fig.colorbar(im, ax=a3, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    fig.suptitle("GPT-2 small (124M) on 10B FineWeb-Edu: induction-head formation",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", default="/mnt/localssd/gpt2/checkpoints")
    ap.add_argument("--out-dir", default="/mnt/localssd/gpt2/induction")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        data = json.load(open(os.path.join(args.out_dir, "induction_results.json")))
    else:
        data = run(args)
    plot(data, os.path.join(args.out_dir, "induction_formation.png"))


if __name__ == "__main__":
    main()
