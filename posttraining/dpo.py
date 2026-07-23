#!/usr/bin/env python
"""
Stage 2 — Direct Preference Optimization on top of the SFT model.

DPO nudges the SFT model toward preferred responses using (prompt, chosen,
rejected) triples, with no reward model or PPO. Uses UltraFeedback, formatted
with the same Alpaca prompt as SFT.

    python posttraining/dpo.py --sft /mnt/localssd/sft/sft_model --out /mnt/localssd/sft/dpo_model
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/mnt/localssd/sft/sft_model")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    ap.add_argument("--out", default="/mnt/localssd/sft/dpo_model")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-samples", type=int, default=15000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def to_pref(ex):
        # UltraFeedback: prompt (str), chosen/rejected are lists of messages.
        return {
            "prompt": build_prompt(ex["prompt"]),
            "chosen": ex["chosen"][-1]["content"],
            "rejected": ex["rejected"][-1]["content"],
        }

    ds = ds.map(to_pref, remove_columns=ds.column_names)
    print(f"DPO pairs: {len(ds)} | example prompt:\n{ds[0]['prompt'][:120]!r}")

    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.pad_token = tok.eos_token

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_length,
        truncation_mode="keep_end",
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = DPOTrainer(model=args.sft, ref_model=None, args=cfg,
                         train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved DPO model -> {args.out}")


if __name__ == "__main__":
    main()
