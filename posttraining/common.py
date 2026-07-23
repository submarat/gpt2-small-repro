"""Shared prompt format for SFT + DPO of the FineWeb-Edu GPT-2 small model.

We use a simple Alpaca-style instruction template (no new special tokens, works
with the stock GPT-2 BPE). The exact same PROMPT is used everywhere so the model
sees a consistent format at SFT time, DPO time, and inference.
"""

BASE_MODEL = "submarat/gpt2-small-fineweb-edu-10b"

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction.strip(), input=input_text.strip())
    return PROMPT.format(instruction=instruction.strip())
