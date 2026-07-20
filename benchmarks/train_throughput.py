#!/usr/bin/env python
"""
End-to-end training throughput on real data.

Runs the actual optimized training path (bf16 autocast + SDPA + torch.compile +
fused AdamW + F.cross_entropy + weight tying + gradient accumulation) over a real
dataset loaded with the repo's own loaders, and reports tokens/sec measured across
whole optimizer steps (after warmup, so compile time is excluded).

This is the number that matters for planning an OpenWebText / FineWeb run: it
includes dataloader overhead, unlike benchmark.py's synthetic model-only harness.

Example:
    python benchmarks/train_throughput.py --dataset wikitext \
        --batch 32 --grad-accum 16 --warmup-steps 3 --measure-steps 10
"""
import argparse
import time

import torch as t
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformer_lens.utils import tokenize_and_concatenate

from gpt2_small import Transformer, TransformerConfig, device


def build_dataloader(name, cfg, batch):
    # Use canonical namespaced repo ids (legacy bare ids fail on newer
    # huggingface_hub). Data content is irrelevant to throughput.
    repo, config_name = {
        "wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    }.get(name, (name, None))
    dataset = load_dataset(repo, config_name, split="train")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tokenized = tokenize_and_concatenate(
        dataset, tok, streaming=False, max_length=cfg.seq_len,
        column_name="text", add_bos_token=True, num_proc=4,
    )
    # Materialize a single [N, seq_len] tensor. Slicing a torch-formatted
    # dataset returns a dict of tensors (indexing a column gives a Column).
    tokens_tensor = tokenized.with_format("torch")[:]["tokens"]
    tensor_ds = t.utils.data.TensorDataset(tokens_tensor)
    return DataLoader(tensor_ds, batch_size=batch, shuffle=True, drop_last=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--batch", type=int, default=32, help="micro-batch (sequences)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup-steps", type=int, default=3, help="optimizer steps (untimed)")
    ap.add_argument("--measure-steps", type=int, default=10, help="optimizer steps (timed)")
    args = ap.parse_args()

    t.set_float32_matmul_precision("high")
    cfg = TransformerConfig(seq_len=args.seq_len)  # full GPT-2 small, sdpa+tie on
    model = Transformer(cfg).to(device)
    model.config = cfg
    n_params = sum(p.numel() for p in model.parameters())
    train_model = t.compile(model)
    optim = t.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == "cuda"))

    dataloader = build_dataloader(args.dataset, cfg, args.batch)
    tokens_per_micro = args.batch * cfg.seq_len
    tokens_per_step = tokens_per_micro * args.grad_accum
    total_steps = args.warmup_steps + args.measure_steps

    print(
        f"{t.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'} | "
        f"params={n_params:,} | micro-batch={args.batch} x seq {cfg.seq_len} | "
        f"grad_accum={args.grad_accum} -> {tokens_per_step:,} tokens/opt-step"
    )

    step = 0
    micro = 0
    t_start = None
    timed_tokens = 0
    data_iter = iter(dataloader)
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats()

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        toks = batch[0].to(device)

        if micro == 0:
            optim.zero_grad(set_to_none=True)
        with t.autocast(device_type=device.type, dtype=t.bfloat16, enabled=device.type == "cuda"):
            logits = train_model(toks)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                toks[:, 1:].reshape(-1),
            ) / args.grad_accum
        loss.backward()
        micro += 1
        if micro == args.grad_accum:
            optim.step()
            micro = 0
            step += 1
            if step == args.warmup_steps:  # start timing after warmup
                if device.type == "cuda":
                    t.cuda.synchronize()
                t_start = time.perf_counter()
            elif step > args.warmup_steps:
                timed_tokens += tokens_per_step

    if device.type == "cuda":
        t.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    tps = timed_tokens / elapsed
    peak_gb = t.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    step_s = elapsed / args.measure_steps

    print("-" * 60)
    print(f"measured over {args.measure_steps} optimizer steps ({args.grad_accum} micro-batches each)")
    print(f"  tokens/sec        : {tps:,.0f}")
    print(f"  sec / opt-step    : {step_s:.2f}")
    print(f"  peak mem (GB)     : {peak_gb:.2f}")
    # Projections
    for label, ntok in [("9B (1 epoch OWT)", 9e9), ("300B (GPT-2 grade)", 300e9)]:
        hrs = ntok / tps / 3600
        print(f"  ETA {label:<20}: {hrs:,.1f} h ({hrs/24:,.1f} days)")


if __name__ == "__main__":
    main()
