import numpy as np
import os
import argparse
from typing import List, Dict
import random

def generate_dummy_data(output_path: str, num_samples: int = 1000, max_len: int = 200, vocab_size: int = 1000):
    """
    Generates dummy training data for Transformer.
    Format:
    - states: [N, max_len] (int64)
    - policies: [N, action_space] (float32)
    - values: [N, 1] (float32)
    """

    # Special Tokens
    PAD = 0
    CLS = 1
    SEP = 2
    MASK = 3
    GLOBAL_STATE_START = 4

    # Generate random sequences
    # Structure: [CLS] [GLOBAL] [SEP] [Card Tokens...] [PAD...]

    states = []
    policies = []
    values = []

    action_space = 600 # Fixed action size as per memory

    print(f"Generating {num_samples} samples...")

    for _ in range(num_samples):
        # 1. State Sequence
        # Global state tokens (e.g. turn, mana count, shields) - simple random integers
        global_features = [random.randint(10, 100) for _ in range(5)]

        # Card tokens (Hand, Mana, Battle) - random card IDs
        # 30 cards in deck, maybe 10 visible
        num_cards = random.randint(5, 30)
        card_tokens = [random.randint(10, vocab_size - 1) for _ in range(num_cards)]

        # Construct sequence
        # Q2: CLS token at start
        seq = [CLS] + global_features + [SEP] + card_tokens

        # Pad or Truncate
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [PAD] * (max_len - len(seq))

        states.append(seq)

        # 2. Policy (Target)
        # Random probability distribution
        p = np.random.dirichlet(np.ones(action_space), size=1)[0]
        policies.append(p)

        # 3. Value (Target)
        # Random value between -1 and 1
        v = np.random.uniform(-1, 1)
        values.append([v])

    # Convert to numpy
    states_np = np.array(states, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.float32)
    values_np = np.array(values, dtype=np.float32)

    # Save
    np.savez_compressed(output_path, states=states_np, policies=policies_np, values=values_np)
    print(f"Saved to {output_path}")
    print(f"Shapes: States={states_np.shape}, Policies={policies_np.shape}, Values={values_np.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/transformer_training_data_dummy.npz")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    generate_dummy_data(args.output, args.samples)
