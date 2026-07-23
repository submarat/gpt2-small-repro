#!/usr/bin/env python
"""
Generate responses from one or more models on a fixed set of instructions, so
we can compare base -> SFT -> DPO behavior.

    python posttraining/chat.py --models base=submarat/gpt2-small-fineweb-edu-10b \
        sft=/mnt/localssd/sft/sft_model dpo=/mnt/localssd/sft/dpo_model
"""
import argparse

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_prompt

PROMPTS = [
    ("List three tips for staying focused while studying.", ""),
    ("Explain what photosynthesis is in one sentence.", ""),
    ("Write a short motivational quote about learning.", ""),
    ("What is the capital of France?", ""),
    ("Summarize the following text.", "The sun is a star at the center of the solar system. It is a nearly perfect ball of hot plasma."),
]


def load(spec):
    name, path = spec.split("=", 1)
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=t.bfloat16).to("cuda").eval()
    return name, model, tok


@t.no_grad()
def respond(model, tok, instruction, input_text):
    prompt = build_prompt(instruction, input_text)
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(
        ids, max_new_tokens=80, do_sample=True, top_k=40, temperature=0.7,
        repetition_penalty=1.3, pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    # Stop at the next instruction marker if the model runs on.
    return text.split("### Instruction")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path_or_hub_id ...")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [load(s) for s in args.models]
    for instruction, input_text in PROMPTS:
        print("=" * 70)
        print("INSTRUCTION:", instruction, f"| INPUT: {input_text}" if input_text else "")
        for name, model, tok in models:
            t.manual_seed(args.seed)
            print(f"\n[{name}] {respond(model, tok, instruction, input_text)}")
        print()


if __name__ == "__main__":
    main()
