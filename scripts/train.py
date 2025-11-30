import os
import sys
import yaml
import glob

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import numpy as np
import dm_ai_module
from py_ai.agent.network import AlphaZeroNetwork
from py_ai.agent.replay_buffer import ReplayBuffer

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'train_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
TRAIN_CFG = CONFIG['training']

def fast_forward(state, card_db):
    dm_ai_module.PhaseManager.fast_forward(state, card_db)



def self_play(network, card_db):
    game_data = []
    gs = dm_ai_module.GameState(np.random.randint(100000))
    gs.setup_test_duel() # Use test deck for now
    dm_ai_module.PhaseManager.start_game(gs, card_db)
    
    mcts_cfg = CONFIG['mcts']
    # Use C++ MCTS
    mcts = dm_ai_module.MCTS(
        card_db, 
        mcts_cfg.get('cpuct', 1.0), # Config might use different key or default
        mcts_cfg['dirichlet_alpha'],
        mcts_cfg['dirichlet_epsilon'],
        TRAIN_CFG.get('mcts_batch_size', 8)
    )
    
    # Evaluator callback for C++ MCTS
    device = next(network.parameters()).device # Get device from network
    
    def evaluator(states):
        # states is list[GameState]
        tensors = []
        for s in states:
            t = dm_ai_module.TensorConverter.convert_to_tensor(s, s.active_player_id, card_db)
            tensors.append(t)
        
        # Stack tensors
        if not tensors:
            return [], []
            
        batch_tensor = torch.tensor(np.array(tensors), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            policy_logits, values = network(batch_tensor)
            
        # Return (list[list[float]], list[float])
        return policy_logits.cpu().tolist(), values.cpu().squeeze(1).tolist()
    
    turn_count = 0
    while True:
        turn_count += 1
        if turn_count > 200: # Safety
            break
            
        # Fast Forward to decision point
        fast_forward(gs, card_db)
        
        # Check Game Over after fast forward
        is_over, result = dm_ai_module.PhaseManager.check_game_over(gs)
        if is_over:
            return game_data, result

        # Temperature Schedule
        if turn_count < 30:
            temp = 1.0
        else:
            temp = 0.1 # Almost deterministic

        # MCTS Search
        # Returns policy vector (std::vector<float>)
        policy = mcts.search(gs, TRAIN_CFG['simulations'], evaluator, add_noise=True, temperature=temp)
        policy = np.array(policy)
                
        # Store data
        # State needs to be converted to tensor
        state_tensor = dm_ai_module.TensorConverter.convert_to_tensor(gs, gs.active_player_id, card_db)
        # Store (state, policy, value_placeholder, active_player_id)
        game_data.append([state_tensor, policy, None, gs.active_player_id]) 
        
        # Select Action
        legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
        
        if not legal_actions:
            break # Should not happen if game not over

        action_probs = []
        for action in legal_actions:
            idx = dm_ai_module.ActionEncoder.action_to_index(action)
            if 0 <= idx < len(policy):
                action_probs.append(policy[idx])
            else:
                action_probs.append(0.0)
        
        action_probs = np.array(action_probs)
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        else:
            # Fallback (uniform)
            action_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
        chosen_idx = np.random.choice(len(legal_actions), p=action_probs)
        action = legal_actions[chosen_idx]
        
        # Apply Action
        dm_ai_module.EffectResolver.resolve_action(gs, action, card_db)
        if action.type == dm_ai_module.ActionType.PASS:
            dm_ai_module.PhaseManager.next_phase(gs, card_db)
            
        # Check Game Over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(gs)
        if is_over:
            # Result: 1=P1_WIN, 2=P2_WIN, 3=DRAW
            return game_data, result
            
    return game_data, 0 # Draw/Timeout

def train_loop():
    print("Starting Training Loop...")
    print(f"Config: {TRAIN_CFG}")
    
    # Load DB
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cards.csv')
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
                network.load_state_dict(torch.load(latest_model, map_location=device))
                start_iteration = latest_iter + 1
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    for iteration in range(start_iteration, start_iteration + iterations):
        print(f"Iteration {iteration}/{start_iteration + iterations - 1}")
        
        # Self Play
        network.eval()
        # Move to CPU for MCTS if needed, or keep on GPU if MCTS supports it
        # Current MCTS implementation uses network(tensor), so it should match device
        
        for i in range(games_per_iter):
            print(f"  Self-play game {i+1}...")
            # Note: self_play runs on CPU logic but uses network. 
            # If network is on GPU, we need to handle tensor device in MCTS or here.
            # MCTS.search calls network.
            # Let's ensure MCTS handles device or we pass device to MCTS?
            # MCTS currently: tensor_t = torch.tensor(...).unsqueeze(0) -> network(tensor_t)
            # We need to move tensor_t to device.
            # For now, let's keep network on CPU for simplicity unless we update MCTS.
            # Or update MCTS to check network device.
            
            # Quick fix: Move network to CPU for self-play if MCTS doesn't handle device
            # But we want GPU for inference.
            # Let's update MCTS later. For now, assume CPU or update MCTS.
            # Actually, let's force CPU for now to be safe as MCTS.py doesn't have .to(device)
            # network.cpu() 
            data, result = self_play(network, card_db)
            
            # Assign Values
            for item in data:
                player_id = item[3]
                value = 0.0
                if result == 1: value = 1.0 if player_id == 0 else -1.0
                elif result == 2: value = 1.0 if player_id == 1 else -1.0
                item[2] = value
                
            cleaned_data = [item[:3] for item in data]
            replay_buffer.push(cleaned_data)
            
        # Training
        print("  Training...")
        network.to(device) # Move back to device for training
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
