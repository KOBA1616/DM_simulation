
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="training_data.npz")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print(f"Error: Cards file not found at {cards_path}")
        sys.exit(1)

    print("Loading card database...")
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    print(f"Initializing DataCollector with {len(card_db)} cards...")
    collector = dm_ai_module.DataCollector(card_db)

    print(f"Collecting {args.episodes} episodes using C++ Heuristic Agent...")
    start_time = time.time()

    # Collect data using the C++ collector which returns tokens and masks
    batch = collector.collect_data_batch_heuristic(args.episodes)

    duration = time.time() - start_time
    print(f"Collection finished in {duration:.2f}s")
    print(f"Collected {len(batch.states)} samples.")

    if len(batch.states) == 0:
        print("No data collected.")
        sys.exit(0)

    # Convert to numpy arrays
    # batch.states is list of lists of int (tokens) -> object array for jagged sequences
    # Convert inner lists to numpy arrays first to ensure compatibility
    tokens = np.array([np.array(s, dtype=np.int64) for s in batch.states], dtype=object)
    policies = np.array(batch.policies, dtype=np.float32)
    masks = np.array(batch.masks, dtype=np.float32)
    values = np.array(batch.values, dtype=np.float32)

    print(f"Saving to {args.output}...")
    np.savez_compressed(args.output,
        tokens=tokens,
        policies=policies,
        masks=masks,
        values=values
    )
    print("Done.")
