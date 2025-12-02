import os
import sys
import yaml
import glob
import csv
import time
from datetime import datetime

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

class TrainingLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Loss', 'WinRate_vs_Best', 'Buffer_Size', 'Time_Sec'])
            
    def log(self, iteration, loss, win_rate, buffer_size, duration):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, f"{loss:.4f}", f"{win_rate:.2f}", buffer_size, f"{duration:.1f}"])
        print(f"  [Log] Iter: {iteration}, Loss: {loss:.4f}, WinRate: {win_rate:.2f}, Time: {duration:.1f}s")

def create_evaluator(network, card_db, device):
    def evaluator(states):
        if not states:
            return [], []
        
        # Use C++ batch conversion
        flat_data = dm_ai_module.TensorConverter.convert_batch_flat(states, card_db)
        input_size = dm_ai_module.TensorConverter.INPUT_SIZE
        
        # Create tensor directly from flat data
        batch_tensor = torch.tensor(flat_data, dtype=torch.float32).view(-1, input_size).to(device)
        
        with torch.no_grad():
            policy_logits, values = network(batch_tensor)
            
        return policy_logits.cpu().tolist(), values.cpu().squeeze(1).tolist()
    return evaluator

def parallel_self_play(network, card_db, num_games, device):
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
    
    eval_func = create_evaluator(network, card_db, device)
    
    # Run games
    num_threads = min(num_games, os.cpu_count() or 4)
    results = runner.play_games(initial_states, eval_func, 1.0, True, num_threads)
    
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

def evaluate_vs_best(new_network, best_network, card_db, num_games, device):
    """
    Play New Network (P1) vs Best Network (P2).
    Returns win rate of New Network.
    """
    runner = dm_ai_module.ParallelRunner(
        card_db,
        TRAIN_CFG['simulations'], # Use same sim count for fairness
        TRAIN_CFG.get('mcts_batch_size', 8)
    )
    
    initial_states = []
    for _ in range(num_games):
        gs = dm_ai_module.GameState(np.random.randint(100000))
        gs.setup_test_duel()
        dm_ai_module.PhaseManager.start_game(gs, card_db)
        initial_states.append(gs)
        
    eval_new = create_evaluator(new_network, card_db, device)
    eval_best = create_evaluator(best_network, card_db, device)
    
    # We need a way to tell ParallelRunner to use different evaluators for P1 and P2.
    # The current C++ binding might only support one evaluator for self-play.
    # If so, we need to wrap them.
    
    def combined_evaluator(states):
        try:
            # This is tricky. The C++ runner sends a batch of states.
            # It doesn't explicitly say "this state is for P1".
            # However, the state object has `active_player`.
            
            states_p1 = []
            indices_p1 = []
            states_p2 = []
            indices_p2 = []
            
            for i, s in enumerate(states):
                # Use active_player_id instead of active_player
                if s.active_player_id == 0:
                    states_p1.append(s)
                    indices_p1.append(i)
                else:
                    states_p2.append(s)
                    indices_p2.append(i)
                    
            results = [None] * len(states)
            
            if states_p1:
                p_logits, vals = eval_new(states_p1)
                for idx, p, v in zip(indices_p1, p_logits, vals):
                    results[idx] = (p, v)
                    
            if states_p2:
                p_logits, vals = eval_best(states_p2)
                for idx, p, v in zip(indices_p2, p_logits, vals):
                    results[idx] = (p, v)
                    
            # Unzip
            final_policies = [r[0] for r in results]
            final_values = [r[1] for r in results]
            return final_policies, final_values
        except Exception as e:
            print(f"Error in combined_evaluator: {e}")
            # Return dummy to avoid C++ crash if possible, though it might still be bad
            return [[0.0]*dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE]*len(states), [0.0]*len(states)


    num_threads = min(num_games, os.cpu_count() or 4)
    # Temperature 0 for evaluation (deterministic play)
    results = runner.play_games(initial_states, combined_evaluator, 0.1, False, num_threads)
    
    wins = 0
    for info in results:
        if info.result == dm_ai_module.GameResult.P1_WIN:
            wins += 1
            
    return wins / num_games

def train_loop():
    print("Starting Training Loop...")
    print(f"Config: {TRAIN_CFG}")
    
    # Load DB
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cards.csv')
    card_db = dm_ai_module.CsvLoader.load_cards(data_path)
    
    # Init Networks
    input_size = dm_ai_module.TensorConverter.INPUT_SIZE
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    
    device = CONFIG['resources']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    network = AlphaZeroNetwork(input_size, action_size).to(device)
    best_network = AlphaZeroNetwork(input_size, action_size).to(device)
    best_network.load_state_dict(network.state_dict()) # Init best as current
    
    optimizer = optim.Adam(network.parameters(), lr=TRAIN_CFG['learning_rate'])
    replay_buffer = ReplayBuffer(capacity=TRAIN_CFG.get('buffer_size', 10000))
    logger = TrainingLogger(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'analytics'))
    
    iterations = TRAIN_CFG['iterations']
    games_per_iter = TRAIN_CFG['games_per_iteration']
    batch_size = TRAIN_CFG['batch_size']
    eval_games = TRAIN_CFG.get('eval_games', 10)
    update_threshold = TRAIN_CFG.get('update_threshold', 0.55)
    
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
            print(f"Resuming from iteration {latest_iter}")
            try:
                state_dict = torch.load(latest_model, map_location=device)
                network.load_state_dict(state_dict)
                best_network.load_state_dict(state_dict) # Assume latest was best
                start_iteration = latest_iter + 1
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    for iteration in range(start_iteration, start_iteration + iterations):
        iter_start_time = time.time()
        print(f"Iteration {iteration}/{start_iteration + iterations - 1}")
        
        # 1. Self Play (Best Network generates data)
        # Usually AlphaZero uses the BEST network to generate data, 
        # or the latest network if we don't do evaluation steps.
        # If we do evaluation, we should use the BEST network to generate high quality data?
        # Or use the LATEST to explore?
        # AlphaGo Zero uses the BEST network for self-play data generation.
        
        best_network.eval()
        print(f"  Self-Play ({games_per_iter} games)...")
        new_data = parallel_self_play(best_network, card_db, games_per_iter, device)
        replay_buffer.push(new_data, is_golden=False)
            
        # 2. Training
        print("  Training...")
        network.train()
        avg_loss = 0
        steps = 0
        
        # Train for some epochs or steps?
        # Usually we train on the new data + buffer.
        # Let's do fixed number of batches per iteration.
        train_steps = max(10, len(new_data) // batch_size) 
        
        for _ in range(train_steps):
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
                
                avg_loss += loss.item()
                steps += 1
        
        avg_loss = avg_loss / steps if steps > 0 else 0
        
        # 3. Evaluation
        print(f"  Evaluating (New vs Best, {eval_games} games)...")
        network.eval()
        win_rate = evaluate_vs_best(network, best_network, card_db, eval_games, device)
        print(f"  Win Rate: {win_rate:.2f}")
        
        if win_rate >= update_threshold:
            print("  -> New Best Model!")
            best_network.load_state_dict(network.state_dict())
            # Save Best Model
            best_model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(best_network.state_dict(), best_model_path)
            
            # Mark recent data as Golden? 
            # Complex to track which data came from which model in this simple loop.
            # For now, just rely on the fact that future self-play will use this better model.
        else:
            print("  -> Rejected.")
            # Revert network to best? 
            # AlphaZero keeps training the same network but only updates the "Self-Play Network" (Best) when it wins.
            # So we DON'T revert `network`. We just don't update `best_network`.
            
        # 4. Checkpointing & Logging
        model_path = os.path.join(models_dir, f"model_iter_{iteration}.pth")
        torch.save(network.state_dict(), model_path)
        
        duration = time.time() - iter_start_time
        logger.log(iteration, avg_loss, win_rate, len(replay_buffer), duration)
        
        # Cleanup
        model_files = glob.glob(os.path.join(models_dir, "model_iter_*.pth"))
        model_files.sort(key=get_iter, reverse=True)
        if len(model_files) > 3:
            for f in model_files[3:]:
                try:
                    os.remove(f)
                except OSError: pass

if __name__ == "__main__":
    train_loop()
