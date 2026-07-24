#!/usr/bin/env python
"""
Multi-turn chat SFT.

Unlike sft.py (single-turn Alpaca), this fine-tunes on a *conversational*
dataset (smol-smoltalk, built for small models) using a ChatML-style chat
template and assistant_only_loss — so the loss is computed on every assistant
turn across a multi-turn conversation, and the model learns turn boundaries.

Starts from the pretrained base (not the Alpaca SFT) so the chat template is
consistent throughout.

    python posttraining/sft_chat.py --out /mnt/localssd/sft/chat_model
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import BASE_MODEL

# ChatML-style template. Role markers are plain text (no new vocab); each turn
# ends with EOS so the model learns to stop. The {% generation %} block marks
# assistant tokens for assistant_only_loss.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "<|assistant|>\n{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n"
    "{% else %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}{{ eos_token }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--out", default="/mnt/localssd/sft/chat_model")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=40000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    # Keep only the messages column (SFTTrainer auto-detects conversational data).
    keep = "messages" if "messages" in ds.column_names else ds.column_names[0]
    ds = ds.select_columns([keep])
    if keep != "messages":
        ds = ds.rename_column(keep, "messages")
    print(f"chat SFT examples: {len(ds)} | turns in example 0: {len(ds[0]['messages'])}")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE

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
        assistant_only_loss=True,   # loss on assistant turns only, across the conversation
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=args.base, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved chat model -> {args.out}")


if __name__ == "__main__":
    main()
