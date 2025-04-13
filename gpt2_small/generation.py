"""
Text generation utilities for the gpt2_small package.

This module provides functions for generating text from a trained transformer model.
"""

import torch as t
from transformers import GPT2TokenizerFast

from .model import device


def generate(model, prompt="Grab the", max_tokens=50, temperature=1.0, top_k=None):
    """
    Generate text from the model.
    
    Args:
        model: Transformer model to use for generation
        prompt: Text prompt to start generation from
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        top_k: If set, only sample from the top k most likely tokens
        
    Returns:
        Generated text as a string
    """
    with t.no_grad():
        # Get config from model
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Model must have a config attribute")
            
        # Get or create tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            tokenizer = model.tokenizer
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the prompt
        tokens = tokenizer(
            prompt, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=config.seq_len,
            padding_side='right',
        )['input_ids']
        tokens = tokens.to(device)

        # Find the first pad token
        seq_len = (tokens[0] == tokenizer.pad_token_id).nonzero()
        if len(seq_len) > 0:
            seq_len = seq_len[0].item()
        else:
            seq_len = len(tokens[0])

        # Generate new tokens up to max_tokens or until sequence length
        for n in range(seq_len, min(seq_len + max_tokens, config.seq_len)):
            logits = model(tokens)
            
            # Get next token logits
            next_token_logits = logits[0, n-1]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = t.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = float('-inf')
            
            # Sample from the distribution
            probs = t.softmax(next_token_logits, dim=0)
            next_token = t.multinomial(probs, 1)
            
            # Replace pad token with new prediction
            tokens[0, n] = next_token
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode the generated tokens
        output_text = tokenizer.decode(tokens[0][:n+1])
        return output_text


def predict(model, input_text, return_logits=False):
    """
    Predict the next tokens for a given input text.
    
    Args:
        model: Transformer model to use for prediction
        input_text: Text input to predict from
        return_logits: Whether to return the raw logits
        
    Returns:
        Predicted text as a string, or (predicted_text, logits) if return_logits=True
    """
    with t.no_grad():
        # Get config from model
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Model must have a config attribute")
            
        # Get or create tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            tokenizer = model.tokenizer
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the input text
        tokens = tokenizer(
            input_text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=config.seq_len,
            padding_side='right',
        )['input_ids']
        tokens = tokens.to(device)

        # Get model predictions
        logits = model(tokens)
        
        # Get the most likely next tokens
        predicted_tokens = logits[0].argmax(dim=-1)
        predicted_text = tokenizer.decode(predicted_tokens)
        
        if return_logits:
            return predicted_text, logits
        else:
            return predicted_text 