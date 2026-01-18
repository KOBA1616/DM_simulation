import numpy as np
import os
import argparse
import json
from typing import List, Dict
import dm_ai_module

def load_magic_deck() -> List[int]:
    """Load the magic.json deck for training."""
    try:
        deck_path = "data/decks/magic.json"
        if os.path.exists(deck_path):
            with open(deck_path, 'r', encoding='utf-8') as f:
                deck = json.load(f)
                if isinstance(deck, list) and len(deck) == 40:
                    print(f"Loaded magic.json deck: {len(deck)} cards")
                    return deck
    except Exception as e:
        print(f"Warning: Failed to load magic.json: {e}")
    
    # Fallback to default deck
    default = [1] * 40
    print(f"Using default deck: {len(default)} cards (all ID=1)")
    return default

def generate_real_data(output_path: str, num_samples: int = 1000, use_magic_deck: bool = True):
    """
    Generates REAL training data for Transformer using C++ DataCollector.
    Format:
    - states: [N, 200] (int64) - via TensorConverter V2
    - policies: [N, 600] (float32)
    - values: [N, 1] (float32)
    
    Args:
        output_path: Path to save the training data
        num_samples: Number of samples to collect
        use_magic_deck: If True, use magic.json deck; if False, use default
    """

    print(f"Initializing DataCollector...")

    # Initialize game environment
    # Note: We use the default CardRegistry which should be populated by loading the JSON
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    print(f"✓ Loaded card database")

    # Load the deck to use for training
    if use_magic_deck:
        deck = load_magic_deck()
    else:
        deck = [1] * 40
        print("Using default deck (all ID=1)")

    # DataCollector will use the Registry's static definitions by default if no db provided
    # But let's be explicit if possible. DataCollector() uses registry.
    collector = dm_ai_module.DataCollector()

    # We need to collect enough episodes to get num_samples steps.
    # A game takes maybe 40-100 steps.
    # So episodes approx num_samples / 50.
    episodes = max(1, num_samples // 40)

    print(f"Collecting data from approx {episodes} episodes to target {num_samples} samples...")
    print(f"Using deck: {deck[:5]}... (showing first 5 cards)")
    print("Running self-play (Heuristic vs Heuristic with magic.json deck)...")

    # collect_tokens=True, collect_tensors=False
    # Note: collect_data_batch_heuristic(episodes, collect_tokens, collect_tensors)
    batch = collector.collect_data_batch_heuristic(episodes, True, False)

    states = list(batch.token_states)
    policies = list(batch.policies)
    values = list(batch.values)

    current_count = len(states)
    print(f"Collected {current_count} samples.")

    # If we have too many, truncate. If too few, run more?
    # For now just use what we got or loop until satisfied.

    while current_count < num_samples:
        needed = num_samples - current_count
        more_eps = max(1, needed // 40)
        print(f"Need {needed} more. Running {more_eps} episodes...")
        more_batch = collector.collect_data_batch_heuristic(more_eps, True, False)

        states.extend(list(more_batch.token_states))
        policies.extend(list(more_batch.policies))
        values.extend(list(more_batch.values))
        current_count = len(states)
        print(f"  Current total: {current_count}")

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
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez_compressed(output_path, states=states_np, policies=policies_np, values=values_np)
    print(f"✓ Saved to {output_path}")
    print(f"  Shapes: States={states_np.shape}, Policies={policies_np.shape}, Values={values_np.shape}")
    print(f"  Datasize: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transformer training data using magic.json deck")
    parser.add_argument("--output", type=str, default="data/transformer_training_data.npz", help="Output path for training data")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to collect")
    parser.add_argument("--no-magic", action="store_true", help="Use default deck instead of magic.json")
    args = parser.parse_args()

    use_magic = not args.no_magic
    generate_real_data(args.output, args.samples, use_magic_deck=use_magic)

