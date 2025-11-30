import os
import sys
import yaml
import glob

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build')))

import torch
import torch.optim as optim
import numpy as np
import dm_ai_module
from py_ai.agent.network import AlphaZeroNetwork
from py_ai.agent.replay_buffer import ReplayBuffer

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'train_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
TRAIN_CFG = CONFIG['training']

def fast_forward(state, card_db):
    dm_ai_module.PhaseManager.fast_forward(state, card_db)



def parallel_self_play(network, card_db, num_games):
    # Setup ParallelRunner
    runner = dm_ai_module.ParallelRunner(
        card_db,
        TRAIN_CFG['simulations'],
        TRAIN_CFG.get('mcts_batch_size', 8)
    )
    
    initial_states = []
    for _ in range(num_games):
        gs = dm_ai_module.GameState(np.random.randint(100000))
        gs.setup_test_duel()
        dm_ai_module.PhaseManager.start_game(gs, card_db)
        initial_states.append(gs)
    
    device = next(network.parameters()).device
    
    def evaluator(states):
        if not states:
            return [], []
            
        # Use C++ batch conversion
        flat_data = dm_ai_module.TensorConverter.convert_batch_flat(states, card_db)
        input_size = dm_ai_module.TensorConverter.INPUT_SIZE
        
        # Create tensor directly from flat data
        # Note: This creates a copy. To avoid copy, we'd need buffer protocol support in binding.
        batch_tensor = torch.tensor(flat_data, dtype=torch.float32).view(-1, input_size).to(device)
        
        with torch.no_grad():
            policy_logits, values = network(batch_tensor)
            
        return policy_logits.cpu().tolist(), values.cpu().squeeze(1).tolist()

    # Run games in parallel
    # Use num_threads = num_games for simplicity, or limit to CPU cores
    num_threads = min(num_games, os.cpu_count() or 4)
    results = runner.play_games(initial_states, evaluator, 1.0, True, num_threads)
    
    all_game_data = []
    
    for info in results:
        result = info.result
        game_data = []
        for i in range(len(info.states)):
            s = info.states[i]
            p = info.policies[i]
            pid = info.active_players[i]
            
            t = dm_ai_module.TensorConverter.convert_to_tensor(s, pid, card_db)
            game_data.append([t, np.array(p), None, pid])
            
        # Assign Values
        for item in game_data:
            player_id = item[3]
            value = 0.0
            if result == dm_ai_module.GameResult.P1_WIN: value = 1.0 if player_id == 0 else -1.0
            elif result == dm_ai_module.GameResult.P2_WIN: value = 1.0 if player_id == 1 else -1.0
            item[2] = value
            
        all_game_data.extend([item[:3] for item in game_data])
            
    return all_game_data

def train_loop():
    print("Starting Training Loop...")
    print(f"Config: {TRAIN_CFG}")
    
    # Load DB
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cards.csv')
    card_db = dm_ai_module.CsvLoader.load_cards(data_path)
    
    # Init Network
    input_size = dm_ai_module.TensorConverter.INPUT_SIZE
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    network = AlphaZeroNetwork(input_size, action_size)
    
    device = CONFIG['resources']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    network.to(device)
    
    optimizer = optim.Adam(network.parameters(), lr=TRAIN_CFG['learning_rate'])
    
    replay_buffer = ReplayBuffer()
    
    iterations = TRAIN_CFG['iterations']
    games_per_iter = TRAIN_CFG['games_per_iteration']
    batch_size = TRAIN_CFG['batch_size']
    
    # Resume Logic
    start_iteration = 0
    models_dir = os.path.join(os.path.dirname(__file__), '..', TRAIN_CFG['checkpoint_dir'])
    os.makedirs(models_dir, exist_ok=True)
    
    model_files = glob.glob(os.path.join(models_dir, "model_iter_*.pth"))
    if model_files:
        def get_iter(f):
            try:
                return int(os.path.basename(f).replace("model_iter_", "").replace(".pth", ""))
            except ValueError:
                return -1
        
        latest_model = max(model_files, key=get_iter)
        latest_iter = get_iter(latest_model)
        
        if latest_iter >= 0:
            print(f"Resuming from iteration {latest_iter} (loading {os.path.basename(latest_model)})")
            try:
                state_dict = torch.load(latest_model, map_location=device)
                # Check shape compatibility
                current_dict = network.state_dict()
                compatible = True
                for k, v in state_dict.items():
                    if k in current_dict:
                        if v.shape != current_dict[k].shape:
                            print(f"Shape mismatch for {k}: {v.shape} vs {current_dict[k].shape}")
                            compatible = False
                            break
                
                if compatible:
                    network.load_state_dict(state_dict)
                    start_iteration = latest_iter + 1
                else:
                    print("Model architecture changed. Starting fresh.")
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    for iteration in range(start_iteration, start_iteration + iterations):
        print(f"Iteration {iteration}/{start_iteration + iterations - 1}")
        
        # Self Play
        network.eval()
        
        print(f"  Running {games_per_iter} parallel games...")
        new_data = parallel_self_play(network, card_db, games_per_iter)
        replay_buffer.push(new_data)
            
        # Training
        print("  Training...")
        network.train()
        
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            if batch:
                states, policies, values = batch
                states = torch.tensor(states, dtype=torch.float32).to(device)
                policies = torch.tensor(policies, dtype=torch.float32).to(device)
                values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                pred_policies, pred_values = network(states)
                
                value_loss = ((pred_values - values) ** 2).mean()
                policy_loss = -(policies * torch.log_softmax(pred_policies, dim=1)).sum(dim=1).mean()
                
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                
                print(f"  Loss: {loss.item():.4f}")
                
        # Save Checkpoint
        models_dir = os.path.join(os.path.dirname(__file__), '..', TRAIN_CFG['checkpoint_dir'])
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"model_iter_{iteration}.pth")
        torch.save(network.state_dict(), model_path)
        
        # Cleanup old models (Keep latest 3)
        model_files = glob.glob(os.path.join(models_dir, "model_iter_*.pth"))
        # Sort by iteration number
        def get_iter(f):
            try:
                return int(os.path.basename(f).replace("model_iter_", "").replace(".pth", ""))
            except ValueError:
                return -1
        
        model_files.sort(key=get_iter, reverse=True)
        
        if len(model_files) > 3:
            for f in model_files[3:]:
                try:
                    os.remove(f)
                    print(f"  Removed old model: {os.path.basename(f)}")
                except OSError as e:
                    print(f"  Error removing {f}: {e}")

if __name__ == "__main__":
    train_loop()
