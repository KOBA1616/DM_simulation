import sys
import os
import torch
import torch.nn as nn

# Mock module if not built
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from python.ai.models.transformer_model import DuelTransformer

def verify_transformer_overlap_logic():
    print("Verifying Transformer Logic and Shape Consistency...")

    vocab_size = 4096
    d_model = 64
    max_seq_len = 2048
    model = DuelTransformer(vocab_size=vocab_size, d_model=d_model, num_layers=2, max_seq_len=max_seq_len)
    model.eval()

    # Test 1: Full length sequence
    # Input length = max_seq_len.
    # Logic adds 1 meta token -> max_seq_len + 1.
    # PE has max_len = max_seq_len + 1.
    # Output should be size max_seq_len + 1 (internally) -> pooled to 1.

    batch_size = 2
    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    print(f"Testing Max Length Input: {dummy_input.shape}")

    try:
        policy, value = model(dummy_input)
        print("Max Length Forward Pass: Success")
        assert policy.shape == (batch_size, 600)
    except Exception as e:
        print(f"Max Length Forward Pass Failed: {e}")
        raise

    # Test 2: Overflow length
    # Input length = max_seq_len + 10.
    # PE should truncate x.
    # Src_extended logic should truncate to match x.

    dummy_input_overflow = torch.randint(0, vocab_size, (batch_size, max_seq_len + 10))
    print(f"Testing Overflow Input: {dummy_input_overflow.shape}")

    try:
        policy, value = model(dummy_input_overflow)
        print("Overflow Forward Pass: Success")
        assert policy.shape == (batch_size, 600)
    except Exception as e:
        print(f"Overflow Forward Pass Failed: {e}")
        raise

if __name__ == "__main__":
    verify_transformer_overlap_logic()
