"""
Training utilities for the gpt2_small package.

This module provides functions for training the transformer model,
including data loading and processing.
"""

import dataclasses
import math
import os
import time

import torch as t
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

from .model import TransformerConfig, device


def generate_dataset(seq_len=512):
    """
    Generate a simple dataset for testing.
    
    Args:
        seq_len: Maximum sequence length for tokenization
        
    Returns:
        List of tokenized texts
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process text by breaking it into tokens.",
        "Neural networks have transformed natural language processing.",
        "Deep learning techniques are widely used in computer vision.",
        "Reinforcement learning is a type of machine learning.",
        "Natural language understanding is a challenging task.",
        "Transfer learning helps in improving model performance.",
        "Convolutional neural networks are effective for image recognition.",
        "Generative adversarial networks can create realistic images.",
        "Attention mechanisms improve the performance of neural networks."
    ]

    # Instantiate the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the sample texts with padding_side='right' and pad to max_length
    tokenized_texts = [
        tokenizer(
            text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            padding_side='right'
        ) 
        for text in sample_texts
    ]

    return tokenized_texts


def _load_hf_dataset(repo: str, config_name, split: str):
    """load_dataset that works across datasets versions.

    Tries the standard (parquet) path first; only older, loading-script-based
    datasets on older datasets versions need trust_remote_code, so retry with it
    if and only if the error asks for it.
    """
    try:
        return load_dataset(repo, config_name, split=split)
    except (ValueError, RuntimeError) as exc:
        if 'trust_remote_code' in str(exc):
            return load_dataset(repo, config_name, split=split, trust_remote_code=True)
        raise


def _resolve_tokenizer(reference_tokenizer=None):
    if reference_tokenizer is not None:
        return reference_tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_token_dataloader(
    repo: str,
    config_name,
    seq_len: int,
    batch_size: int,
    reference_tokenizer=None,
    split: str = 'train',
    num_proc: int = 8,
    num_workers: int = 4,
    column_name: str = 'text',
):
    """Download, tokenize+concatenate into fixed-length blocks, and return a
    DataLoader yielding ``(tokens,)`` batches of shape ``[batch_size, seq_len]``.

    Works for small and large datasets: the tokenized data stays in the
    memory-mapped Arrow dataset and is collated lazily (no giant contiguous
    materialization), so OpenWebText / FineWeb-scale corpora don't blow up RAM.
    """
    tokenizer = _resolve_tokenizer(reference_tokenizer)
    dataset = _load_hf_dataset(repo, config_name, split)

    tokenized = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=seq_len,
        column_name=column_name,
        add_bos_token=True,
        num_proc=num_proc,
    ).with_format('torch', columns=['tokens'])

    def collate(rows):
        # Return a 1-tuple so callers can use batch[0] (matches TensorDataset).
        return (t.stack([row['tokens'] for row in rows]),)

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )


def load_tiny_stories_dataset(config: TransformerConfig, batch_size: int = 64, reference_tokenizer=None):
    """Load the TinyStories dataset as a DataLoader of ``(tokens,)`` batches."""
    return _build_token_dataloader(
        'roneneldan/TinyStories', None, config.seq_len, batch_size, reference_tokenizer,
    )


def load_wikitext_dataset(config: TransformerConfig, batch_size: int = 64, reference_tokenizer=None):
    """Load WikiText-2 (raw) as a DataLoader of ``(tokens,)`` batches."""
    return _build_token_dataloader(
        'Salesforce/wikitext', 'wikitext-2-raw-v1', config.seq_len, batch_size, reference_tokenizer,
    )


def load_pile_10k_dataset(batch_size, max_length, reference_tokenizer=None):
    """Load the Pile-10k dataset as a DataLoader of ``(tokens,)`` batches."""
    return _build_token_dataloader(
        'NeelNanda/pile-10k', None, max_length, batch_size, reference_tokenizer,
    )


def load_openwebtext_dataset(config: TransformerConfig, batch_size: int = 64, reference_tokenizer=None, split: str = 'train'):
    """Load OpenWebText (the community WebText replica, closest to GPT-2's data).

    Full corpus is ~40 GB of text / ~9B GPT-2 tokens. Pass e.g. ``split='train[:5%]'``
    for a subset. Requires HF cache on a large volume.
    """
    return _build_token_dataloader(
        'Skylion007/openwebtext', None, config.seq_len, batch_size, reference_tokenizer, split=split,
    )


def load_fineweb_edu_dataset(config: TransformerConfig, batch_size: int = 64, reference_tokenizer=None, config_name: str = 'sample-10BT', split: str = 'train'):
    """Load FineWeb-Edu (modern high-quality CommonCrawl-derived web text).

    Defaults to the 10BT sample used by current GPT-2 speedrun repros. Use
    ``split='train[:N]'`` to take a subset without the full download.
    """
    return _build_token_dataloader(
        'HuggingFaceFW/fineweb-edu', config_name, config.seq_len, batch_size, reference_tokenizer, split=split,
    )


def cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay from max_lr to min_lr.

    This is the standard GPT-2 / nanoGPT schedule: LR ramps up linearly to avoid
    early instability, then decays smoothly, spending most of the budget at a
    gradually shrinking rate.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, lr, weight_decay, betas, fused):
    """AdamW with decoupled weight decay applied only to >=2D tensors.

    Matmul and embedding weights (>=2D) are regularized; biases and LayerNorm
    gains (1D) are not - the standard GPT-2 split. Tied weights appear once
    because nn.Module.parameters() dedupes shared tensors.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"Optimizer: {n_decay:,} params decayed / {n_no_decay:,} not decayed")
    return t.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


@t.no_grad()
def estimate_loss(fwd_model, val_dataloader, eval_iters, use_amp):
    """Mean next-token cross-entropy over `eval_iters` validation batches."""
    losses = []
    it = iter(val_dataloader)
    for _ in range(eval_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_dataloader)
            batch = next(it)
        toks = batch[0].to(device)
        with t.autocast(device_type=device.type, dtype=t.bfloat16, enabled=use_amp):
            logits = fwd_model(toks)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                toks[:, 1:].reshape(-1),
            )
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path, model, optimizer, step, config):
    """Save the (uncompiled) model + optimizer state and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
        },
        path,
    )


def _infinite(dataloader):
    """Yield batches forever, cycling the dataloader across epochs."""
    while True:
        for batch in dataloader:
            yield batch


def train(model, config, use_wandb=True, **kwargs):
    """
    Train the model using the specified dataset.
    
    Args:
        model: Transformer model to train
        config: TransformerConfig object
        use_wandb: Whether to use Weights & Biases for logging
        **kwargs: Additional training parameters
        
    Training parameters (with defaults):
        lr: Peak learning rate after warmup (6e-4)
        min_lr: Final learning rate after cosine decay (lr/10)
        warmup_steps: Linear-warmup optimizer steps (100)
        max_steps: Total optimizer steps. If None, derived from epochs/
            max_iter_per_epoch (token-budget runs set this directly).
        epochs: Fallback number of epochs when max_steps is None (1)
        max_iter_per_epoch: Fallback per-epoch step cap when max_steps is None
        batch_size: Micro-batch size in sequences (64)
        grad_accum_steps: Micro-batches per optimizer step (1). Effective batch
            = batch_size * grad_accum_steps sequences.
        weight_decay: Decoupled weight decay on >=2D params (0.1)
        betas: AdamW betas ((0.9, 0.95))
        max_grad_norm: Gradient-norm clip value; 0 disables (1.0)
        dataset: 'wikitext'|'tiny-stories'|'pile-10k'|'openwebtext'|
            'fineweb-edu'|'sample'
        val_dataloader: Optional DataLoader of (tokens,) batches for val loss
        val_every: Steps between validation evals; 0 disables (0)
        eval_iters: Batches per validation eval (20)
        checkpoint_dir: If set, save checkpoints here
        checkpoint_every: Steps between checkpoints (0 disables)
        log_every: Steps between console/wandb metric logs (10)
        amp/tf32/compile/fused_optim: optimization toggles (True on CUDA)
    """

    # --- hyperparameters -------------------------------------------------
    lr = kwargs.get('lr', 6e-4)
    min_lr = kwargs.get('min_lr', lr / 10)
    warmup_steps = int(kwargs.get('warmup_steps', 100))
    max_steps = kwargs.get('max_steps', None)
    epochs = kwargs.get('epochs', 1)
    max_iter_per_epoch = kwargs.get('max_iter_per_epoch', None)
    batch_size = kwargs.get('batch_size', 64)
    grad_accum_steps = max(1, int(kwargs.get('grad_accum_steps', 1)))
    weight_decay = kwargs.get('weight_decay', 0.1)
    betas = tuple(kwargs.get('betas', (0.9, 0.95)))
    max_grad_norm = kwargs.get('max_grad_norm', 1.0)
    dataset_type = kwargs.get('dataset', 'wikitext')

    val_dataloader = kwargs.get('val_dataloader', None)
    val_every = int(kwargs.get('val_every', 0))
    eval_iters = int(kwargs.get('eval_iters', 20))
    checkpoint_dir = kwargs.get('checkpoint_dir', None)
    checkpoint_every = int(kwargs.get('checkpoint_every', 0))
    log_every = max(1, int(kwargs.get('log_every', 10)))

    # optimization toggles (CUDA-only)
    on_cuda = device.type == 'cuda'
    use_amp = kwargs.get('amp', True) and on_cuda
    use_tf32 = kwargs.get('tf32', True) and on_cuda
    use_compile = kwargs.get('compile', True)
    use_fused_optim = kwargs.get('fused_optim', True) and on_cuda
    if use_tf32:
        t.set_float32_matmul_precision('high')

    # --- data ------------------------------------------------------------
    ref_tok = getattr(model, 'tokenizer', None)
    if dataset_type == 'wikitext':
        dataloader = load_wikitext_dataset(config, batch_size=batch_size, reference_tokenizer=ref_tok)
    elif dataset_type == 'tiny-stories':
        dataloader = load_tiny_stories_dataset(config, batch_size=batch_size, reference_tokenizer=ref_tok)
    elif dataset_type == 'pile-10k':
        dataloader = load_pile_10k_dataset(batch_size, config.seq_len, reference_tokenizer=ref_tok)
    elif dataset_type in ('openwebtext', 'owt'):
        dataloader = load_openwebtext_dataset(config, batch_size=batch_size, reference_tokenizer=ref_tok)
    elif dataset_type == 'fineweb-edu':
        dataloader = load_fineweb_edu_dataset(config, batch_size=batch_size, reference_tokenizer=ref_tok)
    elif dataset_type == 'sample':
        tokenized_texts = generate_dataset(config.seq_len)
        tokens = [text['input_ids'] for text in tokenized_texts]
        tokens_tensor = t.cat(tokens, dim=0)
        simple_dataset = t.utils.data.TensorDataset(tokens_tensor)
        dataloader = DataLoader(simple_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Resolve the total optimizer-step budget.
    if max_steps is None:
        try:
            steps_per_epoch = max(1, len(dataloader) // grad_accum_steps)
        except TypeError:
            steps_per_epoch = 1
        if max_iter_per_epoch:
            steps_per_epoch = min(steps_per_epoch, int(max_iter_per_epoch))
        max_steps = epochs * steps_per_epoch
    max_steps = max(1, int(max_steps))
    warmup_steps = min(warmup_steps, max_steps)

    # --- optimizer + compile --------------------------------------------
    optimizer = configure_optimizer(model, lr, weight_decay, betas, use_fused_optim)
    # Compile for the loop; keep `model` (uncompiled) for state_dict / attrs.
    train_model = t.compile(model) if use_compile else model

    tokens_per_step = batch_size * grad_accum_steps * config.seq_len
    print(
        f"Effective batch: {batch_size} x {grad_accum_steps} accum = "
        f"{batch_size * grad_accum_steps} seqs ({tokens_per_step:,} tokens)/step | "
        f"max_steps={max_steps:,} (~{max_steps * tokens_per_step / 1e9:.2f}B tokens) | "
        f"warmup={warmup_steps} | peak_lr={lr:g} min_lr={min_lr:g}"
    )

    if use_wandb:
        wandb.init(project="gpt2-small-repro", config={
            "dataset": dataset_type, "peak_lr": lr, "min_lr": min_lr,
            "warmup_steps": warmup_steps, "max_steps": max_steps,
            "batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
            "tokens_per_step": tokens_per_step, "weight_decay": weight_decay,
            "betas": betas, "max_grad_norm": max_grad_norm, "seq_len": config.seq_len,
            "n_params": sum(p.numel() for p in model.parameters()),
        })

    # --- training loop (step-driven) ------------------------------------
    data = _infinite(dataloader)
    if on_cuda:
        t.cuda.synchronize()
    t_log = time.perf_counter()
    tokens_since_log = 0

    for step in tqdm(range(1, max_steps + 1), desc="steps"):
        optimizer.zero_grad(set_to_none=True)

        # Accumulate grads over grad_accum_steps micro-batches. Loss is scaled
        # by 1/grad_accum_steps so the total gradient is the mean over the
        # full effective batch.
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            batch_tokens = next(data)[0].to(device, non_blocking=True)
            with t.autocast(device_type=device.type, dtype=t.bfloat16, enabled=use_amp):
                logits = train_model(batch_tokens)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    batch_tokens[:, 1:].reshape(-1),
                ) / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
        tokens_since_log += tokens_per_step

        # Clip, then set this step's LR from the cosine schedule.
        grad_norm = 0.0
        if max_grad_norm and max_grad_norm > 0:
            grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        lr_now = cosine_lr(step, warmup_steps, max_steps, lr, min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr_now
        optimizer.step()

        # --- metrics ---
        if step % log_every == 0 or step == 1:
            if on_cuda:
                t.cuda.synchronize()
            dt = time.perf_counter() - t_log
            tps = tokens_since_log / dt if dt > 0 else 0.0
            print(f"step {step:>6}/{max_steps} | loss {loss_accum:.4f} | lr {lr_now:.2e} "
                  f"| grad_norm {grad_norm:.2f} | {tps:,.0f} tok/s")
            if use_wandb:
                wandb.log({"train/loss": loss_accum, "lr": lr_now,
                           "grad_norm": grad_norm, "tokens_per_sec": tps,
                           "tokens": step * tokens_per_step}, step=step)
            t_log = time.perf_counter()
            tokens_since_log = 0

        # --- validation ---
        if val_dataloader is not None and val_every and step % val_every == 0:
            val_loss = estimate_loss(train_model, val_dataloader, eval_iters, use_amp)
            print(f"  [val] step {step}: loss {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            t_log = time.perf_counter()  # exclude eval time from throughput
            tokens_since_log = 0

        # --- checkpoint ---
        if checkpoint_dir and checkpoint_every and step % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(path, model, optimizer, step, config)
            print(f"  [ckpt] saved {path}")
            t_log = time.perf_counter()
            tokens_since_log = 0

    if checkpoint_dir:
        save_checkpoint(os.path.join(checkpoint_dir, "ckpt_final.pt"), model, optimizer, max_steps, config)
    if use_wandb:
        wandb.finish()

    return model
