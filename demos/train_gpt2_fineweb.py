#!/usr/bin/env python
"""
GPT-2 small (124M) pre-training on FineWeb-Edu.

Defaults target the modern ~10B-token reproduction recipe (matches GPT-2 on
downstream benchmarks, finishes overnight on one H100). Everything is a CLI flag
so the same script runs a quick smoke test on WikiText.

Full run (background):
    python demos/train_gpt2_fineweb.py

Smoke test (fast, cached data):
    python demos/train_gpt2_fineweb.py --dataset wikitext --batch 8 --grad-accum 2 \
        --max-steps 60 --warmup 10 --val-every 20 --ckpt-every 30 \
        --ckpt-dir /tmp/smoke_ckpt

Hyperparameters follow GPT-2 / nanoGPT: peak LR 6e-4 with cosine decay to 6e-5,
linear warmup, AdamW (0.9, 0.95) with decoupled weight decay 0.1 on >=2D params,
gradient clipping at 1.0, and a ~0.5M-token effective batch via grad accumulation.
"""
import argparse

from gpt2_small import Transformer, TransformerConfig, device, train
from gpt2_small.training import load_fineweb_edu_dataset, load_wikitext_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fineweb-edu")
    ap.add_argument("--batch", type=int, default=32, help="micro-batch (sequences)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--total-tokens", type=float, default=10e9, help="token budget (ignored if --max-steps set)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=700)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=500, help="0 disables validation")
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--val-docs", type=int, default=2000, help="held-out docs for FineWeb val slice")
    ap.add_argument("--ckpt-dir", default="/mnt/localssd/gpt2/checkpoints")
    ap.add_argument("--ckpt-every", type=int, default=2000, help="0 disables checkpoints")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = TransformerConfig(seq_len=args.seq_len)  # full GPT-2 small; sdpa + tied
    model = Transformer(cfg).to(device)
    model.config = cfg

    tokens_per_step = args.batch * args.grad_accum * args.seq_len
    max_steps = args.max_steps or int(args.total_tokens // tokens_per_step)

    # Optional held-out validation loader.
    val_dataloader = None
    if args.val_every:
        if args.dataset == "fineweb-edu":
            # Small disjoint slice (negligible overlap vs. the multi-million-doc train split).
            val_dataloader = load_fineweb_edu_dataset(
                cfg, batch_size=args.batch, split=f"train[:{args.val_docs}]"
            )
        else:
            val_dataloader = load_wikitext_dataset(cfg, batch_size=args.batch)

    print(f"device={device} | dataset={args.dataset} | max_steps={max_steps:,}")
    train(
        model, cfg, use_wandb=not args.no_wandb,
        dataset=args.dataset, batch_size=args.batch, grad_accum_steps=args.grad_accum,
        max_steps=max_steps, lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup,
        weight_decay=args.weight_decay, max_grad_norm=args.grad_clip,
        val_dataloader=val_dataloader, val_every=args.val_every, eval_iters=args.eval_iters,
        checkpoint_dir=args.ckpt_dir, checkpoint_every=args.ckpt_every,
        log_every=args.log_every, compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
