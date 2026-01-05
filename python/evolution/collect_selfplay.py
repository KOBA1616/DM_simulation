
import os
import sys
import time
import argparse
import numpy as np
import torch
from typing import Any, List, Optional, Tuple

# Setup Paths
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

# Ensure project root is in path for dm_toolkit import
if project_root not in sys.path:
    sys.path.append(project_root)

from dm_toolkit.ai.agent.network import AlphaZeroTransformer, AlphaZeroNetwork

def register_callback(network: Any, device: torch.device, model_type: str):
    """Registers the batch inference callback for the network."""
    if model_type == "transformer":
        def sequence_batch_inference(token_lists: List[List[int]]) -> Tuple[List[List[float]], List[float]]:
            max_len = 0
            for tokens in token_lists:
                max_len = max(max_len, len(tokens))
            max_len = max(max_len, 1)

            batch_size = len(token_lists)
            padded_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
            mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

            for i, tokens in enumerate(token_lists):
                L = len(tokens)
                if L > 0:
                    t = torch.tensor(tokens, dtype=torch.long, device=device)
                    padded_tensor[i, :L] = t
                    mask_tensor[i, :L] = True

            with torch.no_grad():
                policy_logits, values = network(padded_tensor, mask=mask_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()
                return policies.tolist(), vals.tolist()

        dm_ai_module.set_sequence_batch_callback(sequence_batch_inference)
    else:
        def flat_batch_inference(input_array: Any) -> Tuple[Any, Any]:
            if isinstance(input_array, list):
                    input_array = np.array(input_array, dtype=np.float32)
            tensor = torch.from_numpy(input_array).float().to(device)
            with torch.no_grad():
                policy_logits, values = network(tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()
                return policies, vals

        dm_ai_module.set_flat_batch_callback(flat_batch_inference)

def collect_selfplay_data(
    episodes: int,
    output_path: str,
    model_path: Optional[str],
    model_type: str = "transformer",
    sims: int = 800,
    batch_size: int = 32,
    threads: int = 4,
    meta_decks_path: Optional[str] = None
):
    print(f"Starting Self-Play Collection: {episodes} episodes, Sims={sims}, Model={model_path or 'Random'}")

    # Load Cards
    cards_path = os.path.join(project_root, 'data', 'cards.json')
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Network
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    network: Any = None

    if model_type == "transformer":
        vocab_size = dm_ai_module.TokenConverter.get_vocab_size()
        network = AlphaZeroTransformer(action_size=action_size, vocab_size=vocab_size, max_seq_len=200).to(device)
    else:
        # Dummy instance for input size
        dummy = dm_ai_module.GameInstance(42, card_db)
        if hasattr(dm_ai_module.TensorConverter, "convert_to_tensor"):
            dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy.state, 0, card_db)
            input_size = len(dummy_vec)
        else:
            input_size = 205
        network = AlphaZeroNetwork(input_size, action_size).to(device)

    # Load Weights
    if model_path and os.path.exists(model_path):
        try:
            network.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("Using Random Weights")

    network.eval()
    register_callback(network, device, model_type)

    # Initial States
    initial_states = []
    # Using simple decks (dummy ID 1) via reset_with_scenario isn't ideal for real self-play
    # Ideally, we load decks. For now, use ParallelRunner's play_games which expects initialized states.
    # We create GameInstances and reset them.
    # IMPORTANT: Real self-play needs real decks.
    # We will use PIMC logic or random decks if meta_decks not provided?
    # ParallelRunner has play_games.

    # For robust PBT, we should use `ParallelRunner.play_deck_matchup`?
    # But `play_games` is more generic.
    # Let's use `GameInstance.reset_with_scenario` with a config that specifies decks if possible,
    # OR rely on `GameInstance` default random deck logic if scenario is not set?
    # GameInstance constructor creates empty state.

    # Let's use a default config with PIMC=True if meta_decks provided.

    for i in range(episodes):
        inst = dm_ai_module.GameInstance(int(time.time()*1000 + i) % 1000000, card_db)
        # Just use default reset which gives random decks (or 30 dummy cards).
        # To get real decks, we need to inject them.
        # But for now, sticking to basics to ensure script works.

        # NOTE: GameInstance defaults to 30 dummy cards (ID 0/1) usually.
        # If we want Meta Decks, we should use ParallelRunner's PIMC loading.
        # But ParallelRunner.play_games takes states.

        # Reset with scenario to ensure consistent starting state
        cfg = dm_ai_module.ScenarioConfig() # Defaults
        inst.reset_with_scenario(cfg)
        initial_states.append(inst.state.clone())

    # Runner
    runner = dm_ai_module.ParallelRunner(card_db, sims, batch_size)
    if meta_decks_path and os.path.exists(meta_decks_path):
        runner.enable_pimc(True)
        runner.load_meta_decks(meta_decks_path)

    evaluator = dm_ai_module.NeuralEvaluator(card_db)
    if model_type == "transformer":
        evaluator.set_model_type(dm_ai_module.ModelType.TRANSFORMER)
    else:
        evaluator.set_model_type(dm_ai_module.ModelType.RESNET)

    # Run
    print("Running simulations...")
    start = time.time()

    # Pass collect_data=True
    # Returns list of GameResultInfo
    results_info = runner.play_games(
        initial_states,
        evaluator,
        1.0,   # temp
        True,  # noise
        threads,
        0.0,   # alpha
        True   # COLLECT DATA
    )

    duration = time.time() - start
    print(f"Finished in {duration:.2f}s")

    # Process Results
    # GameResultInfo contains: result, turn_count, states (vec<GameState>), policies, active_players
    # We need to format this for train_simple.py
    # train_simple expects .npz with: policies, values, (tokens OR states), (masks)

    all_policies = []
    all_values = []
    all_tokens = [] # List of lists
    all_states_tensor = [] # List of lists (flat)

    sample_count = 0

    for info in results_info:
        # info.states is vector<GameState>
        # info.policies is vector<vector<float>>
        # info.active_players is vector<int>
        # We need to compute values.
        # Winner:
        # P1_WIN (1) -> P1=+1, P2=-1
        # P2_WIN (2) -> P1=-1, P2=+1
        # DRAW (3) -> 0

        final_result = info.result

        L = len(info.states)
        for t in range(L):
            state = info.states[t] # GameState shared_ptr
            policy = info.policies[t]
            player = info.active_players[t]

            # Compute Value
            val = 0.0
            if final_result == 1: # P1 Win
                val = 1.0 if player == 0 else -1.0
            elif final_result == 2: # P2 Win
                val = -1.0 if player == 0 else 1.0
            else:
                val = 0.0

            all_policies.append(policy)
            all_values.append(val)

            # Convert State
            # Warning: Converting state in Python loop is slow.
            # C++ DataCollector does this efficiently.
            # Ideally ParallelRunner returns pre-converted tensors/tokens.
            # But GameResultInfo returns GameState objects.
            # We must use TensorConverter/TokenConverter here.

            if model_type == "transformer":
                # TokenConverter
                # Need to handle this carefully.
                # Assuming TokenConverter.encode_state returns list[int]
                tokens = dm_ai_module.TokenConverter.encode_state(state, player, 200)
                all_tokens.append(tokens)
            else:
                # TensorConverter
                vec = dm_ai_module.TensorConverter.convert_to_tensor(state, player, card_db)
                all_states_tensor.append(vec)

            sample_count += 1

    if sample_count == 0:
        print("No samples collected.")
        return

    print(f"Collected {sample_count} samples. Saving...")

    save_dict = {
        'policies': np.array(all_policies, dtype=np.float32),
        'values': np.array(all_values, dtype=np.float32)
    }

    if model_type == "transformer":
        # Object array for variable length
        save_dict['tokens'] = np.array([np.array(x, dtype=np.int64) for x in all_tokens], dtype=object)
    else:
        save_dict['states'] = np.array(all_states_tensor, dtype=np.float32)

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved to {output_path}")

    # Cleanup callbacks
    if model_type == "transformer":
        dm_ai_module.clear_sequence_batch_callback()
    else:
        dm_ai_module.clear_flat_batch_callback()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--type", type=str, default="transformer", choices=["transformer", "resnet"])
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--meta_decks", type=str, default=None)

    args = parser.parse_args()
    collect_selfplay_data(args.episodes, args.output, args.model, args.type, args.sims, meta_decks_path=args.meta_decks)
