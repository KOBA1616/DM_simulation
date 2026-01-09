import numpy as np
import os
import argparse
from typing import List, Dict
import dm_ai_module

def generate_real_data(output_path: str, num_samples: int = 1000):
    """
    Generates REAL training data for Transformer using C++ DataCollector.
    Format:
    - states: [N, 200] (int64) - via TensorConverter V2
    - policies: [N, 600] (float32)
    - values: [N, 1] (float32)
    """

    print(f"Initializing DataCollector...")

    # Initialize game environment
    # Note: We use the default CardRegistry which should be populated by loading the JSON
    dm_ai_module.JsonLoader.load_cards("data/cards.json")

    # DataCollector will use the Registry's static definitions by default if no db provided
    # But let's be explicit if possible. DataCollector() uses registry.
    collector = dm_ai_module.DataCollector()

    # We need to collect enough episodes to get num_samples steps.
    # A game takes maybe 40-100 steps.
    # So episodes approx num_samples / 50.
    episodes = max(1, num_samples // 40)

    print(f"Collecting data from approx {episodes} episodes to target {num_samples} samples...")
    print("Running self-play (Heuristic vs Heuristic)...")

    # collect_tokens=True, collect_tensors=False
    # Note: collect_data_batch_heuristic(episodes, collect_tokens, collect_tensors)
    batch = collector.collect_data_batch_heuristic(episodes, True, False)

    states = batch.token_states
    policies = batch.policies
    values = batch.values

    current_count = len(states)
    print(f"Collected {current_count} samples.")

    # If we have too many, truncate. If too few, run more?
    # For now just use what we got or loop until satisfied.

    while current_count < num_samples:
        needed = num_samples - current_count
        more_eps = max(1, needed // 40)
        print(f"Need {needed} more. Running {more_eps} episodes...")
        more_batch = collector.collect_data_batch_heuristic(more_eps, True, False)

        states.extend(more_batch.token_states)
        policies.extend(more_batch.policies)
        values.extend(more_batch.values)
        current_count = len(states)

    # Truncate to exact amount
    states = states[:num_samples]
    policies = policies[:num_samples]
    values = values[:num_samples]

    # Convert to numpy
    print("Converting to numpy...")

    # Pad states to 200 if they aren't already (C++ should handle this, but verify)
    # TensorConverter::convert_to_sequence uses MAX_SEQ_LEN = 200

    states_np = np.array(states, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.float32)
    values_np = np.array(values, dtype=np.float32).reshape(-1, 1)

    # Save
    np.savez_compressed(output_path, states=states_np, policies=policies_np, values=values_np)
    print(f"Saved to {output_path}")
    print(f"Shapes: States={states_np.shape}, Policies={policies_np.shape}, Values={values_np.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/transformer_training_data.npz")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    generate_real_data(args.output, args.samples)
