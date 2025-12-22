
import numpy as np
import os
import argparse

def generate_dummy_data(output_dir="data", samples=1000):
    os.makedirs(output_dir, exist_ok=True)

    # Constants
    INPUT_SIZE = 226  # ResNet State Vector Size (Updated from log)
    ACTION_SIZE = 591 # Action Size (Updated from log)
    VOCAB_SIZE = 1000
    SEQ_LEN = 32

    # 1. Generate ResNet Data (States)
    print(f"Generating {samples} samples for ResNet (States)...")
    states = np.random.rand(samples, INPUT_SIZE).astype(np.float32)
    policies = np.random.rand(samples, ACTION_SIZE).astype(np.float32)
    # Normalize policies to sum to 1
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = np.random.uniform(-1, 1, size=(samples, 1)).astype(np.float32)

    np.savez(
        os.path.join(output_dir, "dummy_resnet.npz"),
        states=states,
        policies=policies,
        values=values
    )

    # 2. Generate Transformer Data (Tokens)
    print(f"Generating {samples} samples for Transformer (Tokens)...")
    # Tokens are jagged arrays usually, but for dummy we can make them fixed or list of arrays
    tokens_list = []
    for _ in range(samples):
        # Random length between 10 and SEQ_LEN
        length = np.random.randint(10, SEQ_LEN)
        t = np.random.randint(0, VOCAB_SIZE, size=(length,), dtype=np.int64)
        tokens_list.append(t)

    # Tokens must be stored as object array for save
    tokens_arr = np.array(tokens_list, dtype=object)

    np.savez(
        os.path.join(output_dir, "dummy_transformer.npz"),
        tokens=tokens_arr,
        policies=policies,
        values=values
    )

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    generate_dummy_data(args.output_dir, args.samples)
