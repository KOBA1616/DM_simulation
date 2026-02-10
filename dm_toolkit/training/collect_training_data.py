
import os
import sys
import numpy as np
from typing import Any, cast
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
    parser.add_argument("--mode", type=str, default="resnet", choices=["resnet", "transformer", "both"],
                        help="Data collection mode: 'resnet' (flat tensors), 'transformer' (tokens), or 'both'")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print(f"Error: Cards file not found at {cards_path}")
        sys.exit(1)

    print("Loading card database...")
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    print(f"Initializing DataCollector with {len(card_db)} cards...")
    collector = dm_ai_module.DataCollector(card_db)

    collect_tokens = (args.mode == "transformer") or (args.mode == "both")
    collect_tensors = (args.mode == "resnet") or (args.mode == "both")

    print(f"Collecting {args.episodes} episodes using C++ Heuristic Agent (Tokens={collect_tokens}, Tensors={collect_tensors})...")
    start_time = time.time()

    # Collect data using the C++ collector
    batch = collector.collect_data_batch_heuristic(args.episodes, collect_tokens, collect_tensors)

    duration = time.time() - start_time
    # Determine sample count based on what was collected
    sample_count = 0
    if collect_tensors and len(batch.tensor_states) > 0:
        sample_count = len(batch.tensor_states)
    elif collect_tokens and len(batch.token_states) > 0:
        sample_count = len(batch.token_states)

    print(f"Collection finished in {duration:.2f}s")
    print(f"Collected {sample_count} samples.")

    if sample_count == 0:
        print("No data collected.")
        sys.exit(0)

    # Convert to numpy arrays
    policies_raw = batch.policies
    masks = np.array(batch.masks, dtype=np.float32)
    values = np.array(batch.values, dtype=np.float32)

    # Normalize policies if they are returned as command-like objects
    try:
        from dm_toolkit.training.command_compat import command_to_index
    except Exception:
        command_to_index = None

    def _is_prob_array(obj):
        try:
            a = np.array(obj, dtype=np.float32)
            return np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer)
        except Exception:
            return False

    if isinstance(policies_raw, (list, tuple)) and len(policies_raw) > 0 and not _is_prob_array(policies_raw[0]):
        # Need to convert command-like policies into canonical one-hot/prob vectors
        new_dim = None
        try:
            if hasattr(dm_ai_module, 'CommandEncoder') and getattr(dm_ai_module.CommandEncoder, 'TOTAL_COMMAND_SIZE', None) is not None:
                new_dim = int(dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE)
        except Exception:
            new_dim = None

        if new_dim is None or command_to_index is None:
            print("Warning: Unable to normalize command-like policies (CommandEncoder or command_compat missing). Saving raw policies as object array.")
            policies = np.array(policies_raw, dtype=object)
        else:
            policies = np.zeros((len(policies_raw), new_dim), dtype=np.float32)
            for i, entry in enumerate(policies_raw):
                # entry may be dict{cmd:prob}, list of cmds, or single cmd
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        idx = command_to_index(k) if not isinstance(k, int) else k
                        if idx is not None and 0 <= idx < new_dim:
                            policies[i, int(idx)] += float(v)
                elif isinstance(entry, (list, tuple)):
                    for item in entry:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            k, v = item
                            idx = command_to_index(k) if not isinstance(k, int) else k
                            if idx is not None and 0 <= idx < new_dim:
                                policies[i, int(idx)] += float(v)
                        else:
                            idx = command_to_index(item) if not isinstance(item, int) else item
                            if idx is not None and 0 <= idx < new_dim:
                                policies[i, int(idx)] += 1.0
                else:
                    idx = command_to_index(entry) if not isinstance(entry, int) else entry
                    if idx is not None and 0 <= idx < new_dim:
                        policies[i, int(idx)] += 1.0

            # Normalize rows
            row_sums = policies.sum(axis=1, keepdims=True)
            mask = row_sums.squeeze() > 0
            if mask.any():
                policies[mask] = policies[mask] / row_sums[mask]
    else:
        policies = np.array(policies_raw, dtype=np.float32)

    save_dict = {
        'policies': policies,
        'masks': masks,
        'values': values
    }

    if collect_tokens:
        # batch.token_states is list of lists of int -> object array for jagged sequences
        tokens = np.array([np.array(s, dtype=np.int64) for s in batch.token_states], dtype=object)
        save_dict['tokens'] = tokens

    if collect_tensors:
        # batch.tensor_states is list of list of float -> matrix
        states = np.array(batch.tensor_states, dtype=np.float32)
        save_dict['states'] = states

    print(f"Saving to {args.output}...")
    np.savez_compressed(args.output, **cast(dict, save_dict))
    print("Done.")
