
import os
import sys
import numpy as np
import argparse
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

def collect_data(episodes, output_file, card_db):
    print(f"Initializing DataCollector for {episodes} episodes...")
    collector = dm_ai_module.DataCollector(card_db)

    start_time = time.time()

    # Run collection in C++
    print(f"Running collection loop in C++...")
    batch = collector.collect_data_batch(episodes)

    duration = time.time() - start_time
    num_samples = len(batch.states)

    print(f"Done. Time: {duration:.2f}s. Samples: {num_samples}")

    if num_samples == 0:
        print("Warning: No samples collected.")
        return

    # Convert to numpy arrays
    # Note: CollectedBatch.states corresponds to the masked states used for training input.
    states = np.array(batch.states, dtype=np.float32)
    policies = np.array(batch.policies, dtype=np.float32)
    values = np.array(batch.values, dtype=np.float32)

    # Save to .npz
    # We use 'states' key now. 'states_full' is no longer available from this C++ collector
    # but train_simple.py has been updated to handle 'states' or 'states_masked'.
    np.savez_compressed(output_file,
        states=states,
        policies=policies,
        values=values
    )
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="training_data.npz")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print(f"Error: cards.json not found at {cards_path}")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    collect_data(args.episodes, args.output, card_db)
