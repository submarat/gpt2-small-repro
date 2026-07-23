#!/usr/bin/env python
"""
Stage 1 — Supervised fine-tuning (instruction tuning).

Turns the FineWeb-Edu base model into a toy instruction-follower by SFT on
Alpaca, formatted as prompt/completion so TRL masks the prompt and trains the
loss on the response only (completion_only_loss).

    python posttraining/sft.py --out /mnt/localssd/sft/sft_model
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import BASE_MODEL, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--out", default="/mnt/localssd/sft/sft_model")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def to_prompt_completion(ex):
        return {
            "prompt": build_prompt(ex["instruction"], ex.get("input", "")),
            "completion": ex["output"],
        }

    ds = ds.map(to_prompt_completion, remove_columns=ds.column_names)
    print(f"SFT examples: {len(ds)} | example prompt:\n{ds[0]['prompt']!r}\n-> {ds[0]['completion'][:80]!r}")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,   # loss on the response only
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=args.base, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved SFT model -> {args.out}")


if __name__ == "__main__":
    main()
