#!/usr/bin/env python
"""
Evaluate a trained gpt2_small checkpoint on standard LM benchmarks via the
EleutherAI lm-evaluation-harness CLI, side-by-side with the public `gpt2`.

Our model is a custom Transformer, not a HuggingFace model, so we first convert
its weights into an equivalent `GPT2LMHeadModel` and save it as a HF model dir.
The lm_eval CLI then evaluates it with `--model hf` exactly like any HF model.

Architecture note: this repo uses exact (erf) GELU, so the converted HF config
sets activation_function="gelu" (not GPT-2's default tanh-approx "gelu_new").
The public `gpt2` reference is evaluated with its own native config.

Usage:
    # Convert + evaluate our checkpoint and compare against public gpt2
    python benchmarks/eval_lm_harness.py --checkpoint /mnt/localssd/gpt2/checkpoints/ckpt_final.pt

    # Just verify the converter is numerically exact (no benchmarks, fast)
    python benchmarks/eval_lm_harness.py --verify-only

    # Quick subset run
    python benchmarks/eval_lm_harness.py --checkpoint ... --limit 200 --tasks hellaswag
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import torch as t

from gpt2_small import Transformer, TransformerConfig, device

DEFAULT_TASKS = "hellaswag,lambada_openai,piqa,arc_easy,sciq,wikitext"


def _load_our_model(checkpoint=None):
    """Rebuild our Transformer from a checkpoint (or random init if None)."""
    if checkpoint is not None:
        ckpt = t.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = TransformerConfig(**ckpt["config"])
        model = Transformer(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        cfg = TransformerConfig()
        model = Transformer(cfg)
    return model.eval(), cfg


def convert_to_hf(model, cfg):
    """Map our Transformer weights into a HuggingFace GPT2LMHeadModel.

    Our per-head projections (W_Q/W_K/W_V: [n_heads, d_model, d_head]) are fused
    into HF's c_attn ([d_model, 3*d_model]); W_O ([n_heads, d_head, d_model]) ->
    c_proj. Our MLP/LayerNorm/embeddings map directly (HF Conv1D uses [in, out]
    weights, matching our x @ W convention).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    d_model, n_heads, d_head = cfg.d_model, cfg.n_heads, cfg.d_head
    hf_cfg = GPT2Config(
        vocab_size=cfg.vocab,
        n_positions=cfg.seq_len,
        n_embd=d_model,
        n_layer=cfg.n_layers,
        n_head=n_heads,
        n_inner=cfg.d_mlp,
        activation_function="gelu",       # exact erf GELU, matches training
        layer_norm_epsilon=cfg.layer_norm_eps,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    hf = GPT2LMHeadModel(hf_cfg).eval()
    sd = {}

    sd["transformer.wte.weight"] = model.embed.W_E.detach().clone()
    sd["transformer.wpe.weight"] = model.pos_embed.W_pos.detach().clone()

    for i, block in enumerate(model.blocks):
        p = f"transformer.h.{i}."
        sd[p + "ln_1.weight"] = block.ln1.w.detach().clone()
        sd[p + "ln_1.bias"] = block.ln1.b.detach().clone()
        sd[p + "ln_2.weight"] = block.ln2.w.detach().clone()
        sd[p + "ln_2.bias"] = block.ln2.b.detach().clone()

        attn = block.attn
        # [n_heads, d_model, d_head] -> [d_model, n_heads*d_head], then fuse QKV.
        wq = attn.W_Q.detach().permute(1, 0, 2).reshape(d_model, n_heads * d_head)
        wk = attn.W_K.detach().permute(1, 0, 2).reshape(d_model, n_heads * d_head)
        wv = attn.W_V.detach().permute(1, 0, 2).reshape(d_model, n_heads * d_head)
        sd[p + "attn.c_attn.weight"] = t.cat([wq, wk, wv], dim=1).clone()
        bq = attn.b_Q.detach().reshape(n_heads * d_head)
        bk = attn.b_K.detach().reshape(n_heads * d_head)
        bv = attn.b_V.detach().reshape(n_heads * d_head)
        sd[p + "attn.c_attn.bias"] = t.cat([bq, bk, bv], dim=0).clone()
        # [n_heads, d_head, d_model] -> [d_model, d_model]
        sd[p + "attn.c_proj.weight"] = attn.W_O.detach().reshape(n_heads * d_head, d_model).clone()
        sd[p + "attn.c_proj.bias"] = attn.b_O.detach().clone()

        mlp = block.mlp
        sd[p + "mlp.c_fc.weight"] = mlp.W_in.detach().clone()
        sd[p + "mlp.c_fc.bias"] = mlp.b_in.detach().clone()
        sd[p + "mlp.c_proj.weight"] = mlp.W_out.detach().clone()
        sd[p + "mlp.c_proj.bias"] = mlp.b_out.detach().clone()

    sd["transformer.ln_f.weight"] = model.ln_final.w.detach().clone()
    sd["transformer.ln_f.bias"] = model.ln_final.b.detach().clone()
    sd["lm_head.weight"] = model.embed.W_E.detach().clone()  # tied

    missing, unexpected = hf.load_state_dict(sd, strict=False)
    # HF may keep buffers (attn bias/masks) as "missing"; no real weights should be.
    real_missing = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    real_missing = [k for k in real_missing if "attn.bias" not in k and "masked_bias" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"Weight mapping incomplete. missing={real_missing} unexpected={unexpected}")
    return hf


@t.no_grad()
def verify_conversion(seq_len=64, atol=2e-4):
    """Convert a random model and check HF logits match ours on the same input."""
    model, cfg = _load_our_model(None)
    model = model.to(device).eval()
    hf = convert_to_hf(model.cpu(), cfg).to(device).eval()
    model = model.to(device)
    tokens = t.randint(0, cfg.vocab, (2, seq_len), device=device)
    ours = model(tokens).float()
    theirs = hf(tokens).logits.float()
    max_diff = (ours - theirs).abs().max().item()
    ok = max_diff < atol
    print(f"[verify] max |logit diff| = {max_diff:.2e}  ->  {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("Converter mismatch - aborting.")
    return ok


def run_lm_eval(model_args, tasks, out_path, batch_size, limit, device_str):
    cmd = [
        "lm_eval", "--model", "hf", "--model_args", model_args,
        "--tasks", tasks, "--device", device_str,
        "--batch_size", str(batch_size), "--output_path", out_path,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_results(out_path):
    files = glob.glob(os.path.join(out_path, "**", "results*.json"), recursive=True)
    if not files:
        return {}
    latest = max(files, key=os.path.getmtime)
    data = json.loads(open(latest).read())
    out = {}
    for task, metrics in data.get("results", {}).items():
        # Prefer acc_norm, fall back to acc / perplexity.
        for key in ("acc_norm,none", "acc,none", "word_perplexity,none", "perplexity,none"):
            if key in metrics:
                out[task] = (key.split(",")[0], metrics[key])
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="/mnt/localssd/gpt2/checkpoints/ckpt_final.pt")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", default="/mnt/localssd/gpt2/eval")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="cap examples/task (quick runs)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-gpt2", action="store_true", help="don't also eval public gpt2")
    ap.add_argument("--gpt2-only", action="store_true", help="eval only public gpt2 (no checkpoint needed)")
    ap.add_argument("--verify-only", action="store_true", help="only run the converter numeric check")
    args = ap.parse_args()

    if args.verify_only:
        verify_conversion()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    if not args.gpt2_only:
        # Sanity-check the converter, then convert our checkpoint to a HF model dir.
        verify_conversion()
        model, cfg = _load_our_model(args.checkpoint)
        hf = convert_to_hf(model, cfg)
        hf_dir = os.path.join(args.out_dir, "hf_model")
        hf.save_pretrained(hf_dir)
        from transformers import GPT2TokenizerFast
        GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(hf_dir)
        print(f"Saved converted HF model -> {hf_dir}")

        ours_out = os.path.join(args.out_dir, "results_ours")
        run_lm_eval(f"pretrained={hf_dir},dtype=bfloat16", args.tasks, ours_out,
                    args.batch_size, args.limit, args.device)
        results["ours (trained)"] = parse_results(ours_out)

    if not args.skip_gpt2:
        gpt2_out = os.path.join(args.out_dir, "results_gpt2")
        run_lm_eval("pretrained=gpt2,dtype=bfloat16", args.tasks, gpt2_out,
                    args.batch_size, args.limit, args.device)
        results["gpt2 (public)"] = parse_results(gpt2_out)

    # Side-by-side summary.
    tasks = sorted({tk for r in results.values() for tk in r})
    print("\n" + "=" * 60)
    header = f"{'task':<18}{'metric':<12}" + "".join(f"{name:>16}" for name in results)
    print(header + "\n" + "-" * len(header))
    for task in tasks:
        metric = next((results[n][task][0] for n in results if task in results[n]), "")
        row = f"{task:<18}{metric:<12}"
        for name in results:
            val = results[name].get(task)
            row += f"{val[1]:>16.4f}" if val else f"{'-':>16}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
