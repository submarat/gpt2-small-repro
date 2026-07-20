"""
Training utilities for the gpt2_small package.

This module provides functions for training the transformer model,
including data loading and processing.
"""

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


def train(model, config, use_wandb=True, **kwargs):
    """
    Train the model using the specified dataset.
    
    Args:
        model: Transformer model to train
        config: TransformerConfig object
        use_wandb: Whether to use Weights & Biases for logging
        **kwargs: Additional training parameters
        
    Training parameters (with defaults):
        lr: Learning rate (1e-3)
        epochs: Number of epochs (50)
        batch_size: Batch size (64)
        weight_decay: Weight decay for optimizer (1e-2)
        max_iter_per_epoch: Maximum iterations per epoch (50)
        dataset: Dataset to use ('wikitext', 'tiny-stories', 'pile-10k', or 'sample')
        grad_accum_steps: Micro-batches accumulated per optimizer step (1).
            Effective batch = batch_size * grad_accum_steps sequences.
        amp: Use bf16 autocast on CUDA (True)
        tf32: Allow TF32 matmuls on CUDA (True)
        compile: Wrap the model in torch.compile (True)
        fused_optim: Use fused AdamW on CUDA (True on CUDA)
    """

    # Hyperparameters with defaults
    lr = kwargs.get('lr', 1e-3)
    epochs = kwargs.get('epochs', 50)
    batch_size = kwargs.get('batch_size', 64)
    weight_decay = kwargs.get('weight_decay', 1e-2)
    max_iter_per_epoch = kwargs.get('max_iter_per_epoch', 50)
    dataset_type = kwargs.get('dataset', 'wikitext')
    # Micro-batches accumulated per optimizer step. Effective batch =
    # batch_size * grad_accum_steps sequences.
    grad_accum_steps = max(1, int(kwargs.get('grad_accum_steps', 1)))

    # Optimization toggles (see OPTIMIZATION.md). Only take effect on CUDA.
    on_cuda = device.type == 'cuda'
    use_amp = kwargs.get('amp', True) and on_cuda
    use_tf32 = kwargs.get('tf32', True) and on_cuda
    use_compile = kwargs.get('compile', True)
    use_fused_optim = kwargs.get('fused_optim', True) and on_cuda

    if use_tf32:
        t.set_float32_matmul_precision('high')

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="transformer_training", config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_iter_per_epoch": max_iter_per_epoch,
            "dataset": dataset_type
        })
        table = wandb.Table(columns=["input", "next_tokens"])

    # Get dataloader based on dataset type
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
        # Create a small in-memory dataset for quick testing
        tokenized_texts = generate_dataset(config.seq_len)
        # Extract just the input_ids from each tokenized text
        tokens = [text['input_ids'] for text in tokenized_texts]
        # Stack them into a tensor
        tokens_tensor = t.cat(tokens, dim=0)
        # Create a simple dataset and dataloader
        simple_dataset = t.utils.data.TensorDataset(tokens_tensor)
        dataloader = DataLoader(simple_dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    optimizer = t.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=use_fused_optim
    )

    # Compile the model for the training loop; keep `model` (uncompiled) for
    # attribute access (tokenizer/config) and as the returned handle. Weights
    # are shared, so training updates the original model in place.
    train_model = t.compile(model) if use_compile else model

    print(
        f"Effective batch: {batch_size} seqs x {grad_accum_steps} accum "
        f"= {batch_size * grad_accum_steps} seqs "
        f"({batch_size * grad_accum_steps * config.seq_len:,} tokens) per optimizer step"
    )

    step = 0          # optimizer steps taken
    micro = 0         # micro-batches accumulated since the last step
    loss_accum = 0.0  # summed full-batch loss over the current accumulation window
    epoch_loss = float('nan')
    for epoch in tqdm(range(epochs)):
        iterations = 0
        for batch_idx, batch_tokens in enumerate(tqdm(dataloader, desc="Batch Progress")):
            # Zero grads at the start of each accumulation window; any partial
            # window left at an epoch boundary is discarded here (never stepped).
            if micro == 0:
                optimizer.zero_grad(set_to_none=True)

            # batch_tokens will be shape [batch_size, seq_len]
            batch_tokens = batch_tokens[0].to(device)  # Move to device

            # Forward + next-token cross-entropy under bf16 autocast.
            # Scale by 1/grad_accum_steps so the accumulated gradient equals the
            # mean over the full effective batch.
            with t.autocast(device_type=device.type, dtype=t.bfloat16, enabled=use_amp):
                logits = train_model(batch_tokens)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    batch_tokens[:, 1:].reshape(-1),
                ) / grad_accum_steps

            loss.backward()
            loss_accum += loss.item()
            micro += 1

            # Take an optimizer step only once a full window is accumulated.
            if micro == grad_accum_steps:
                optimizer.step()
                step += 1
                micro = 0
                epoch_loss = loss_accum  # full-batch mean loss for this step
                if use_wandb:
                    wandb.log({"loss": loss_accum}, step=step)
                loss_accum = 0.0

            if max_iter_per_epoch and iterations > max_iter_per_epoch:
                break
            iterations += 1

        # Log the last full-batch loss for the epoch
        if use_wandb:
            wandb.log({"epoch": epoch, "loss": epoch_loss})

            # Only log examples if we're using wandb
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                tokenizer = model.tokenizer
            else:
                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                
            # Decode the final batch_tokens and logits
            decoded_batch_tokens = tokenizer.decode(batch_tokens[0])
            next_tokens = tokenizer.decode(logits[0,:].argmax(dim=-1))
            table.add_data(decoded_batch_tokens, next_tokens)
        print(f"Epoch {epoch} loss: {epoch_loss}")

    if use_wandb:
        wandb.log({"examples": table})
        wandb.finish()

    return model 
