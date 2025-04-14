"""
Training utilities for the gpt2_small package.

This module provides functions for training the transformer model,
including data loading and processing.
"""

import torch as t
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


def load_wikitext_dataset(config: TransformerConfig, batch_size: int = 64, reference_tokenizer=None):
    """
    Load the WikiText dataset for training.
    
    Args:
        config: TransformerConfig object
        batch_size: Batch size for DataLoader
        reference_tokenizer: Optional reference tokenizer to use
        
    Returns:
        DataLoader for the WikiText dataset
    """
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    if reference_tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = reference_tokenizer

    tokenized_dataset = tokenize_and_concatenate(
        dataset, 
        tokenizer, 
        streaming=False, 
        max_length=config.seq_len, 
        column_name="text", 
        add_bos_token=True, 
        num_proc=4
    )

    # Create DataLoader for batching
    tensor_dataset = t.utils.data.TensorDataset(tokenized_dataset['tokens'])
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_pile_10k_dataset(batch_size, max_length, reference_tokenizer=None):
    """
    Load the Pile-10k dataset for training.
    
    Args:
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length for tokenization
        reference_tokenizer: Optional reference tokenizer to use
        
    Returns:
        DataLoader for the Pile-10k dataset
    """
    dataset = load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    
    tokenized_dataset = tokenize_and_concatenate(
        dataset, 
        reference_tokenizer, 
        streaming=False, 
        max_length=max_length, 
        column_name="text", 
        add_bos_token=True, 
        num_proc=4
    )

    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(
        dataset_dict["train"], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader


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
        dataset: Dataset to use ('wikitext', 'pile-10k', or 'sample')
    """
    
    # Hyperparameters with defaults
    lr = kwargs.get('lr', 1e-3)
    epochs = kwargs.get('epochs', 50)
    batch_size = kwargs.get('batch_size', 64)
    weight_decay = kwargs.get('weight_decay', 1e-2)
    max_iter_per_epoch = kwargs.get('max_iter_per_epoch', 50)
    dataset_type = kwargs.get('dataset', 'wikitext')

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
    if dataset_type == 'wikitext':
        dataloader = load_wikitext_dataset(config, batch_size=batch_size)
    elif dataset_type == 'pile-10k':
        dataloader = load_pile_10k_dataset(batch_size, config.seq_len)
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

    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    for epoch in tqdm(range(epochs)):
        loss = t.ones(1)
        iterations = 0
        for batch_idx, batch_tokens in enumerate(tqdm(dataloader, desc="Batch Progress")):
            step += 1
            # batch_tokens will be shape [batch_size, seq_len]
            batch_tokens = batch_tokens[0].to(device)  # Move to device

            # Calculate loss
            logits = model(batch_tokens)
            log_probs = logits.log_softmax(dim=-1)
            log_probs_for_tokens = log_probs[:, :-1]\
                .gather(dim=-1, index=batch_tokens[:, 1:].unsqueeze(-1)).unsqueeze(-1)
            loss = -log_probs_for_tokens.mean()

            if use_wandb:
                wandb.log({"loss": loss}, step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations > max_iter_per_epoch:
                break
            iterations += 1
        
        # Log the last loss for the epoch
        if use_wandb:
            wandb.log({"epoch": epoch, "loss": loss.item()})

            # Only log examples if we're using wandb
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                tokenizer = model.tokenizer
            else:
                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                
            # Decode the final batch_tokens and logits
            decoded_batch_tokens = tokenizer.decode(batch_tokens[0])
            next_tokens = tokenizer.decode(logits[0,:].argmax(dim=-1))
            table.add_data(decoded_batch_tokens, next_tokens)
        print(f"Epoch {epoch} loss: {loss.item()}")

    if use_wandb:
        wandb.log({"examples": table})
        wandb.finish()

    return model 
