import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import numpy as np
import dm_ai_module
from py_ai.agent.network import AlphaZeroNetwork
from py_ai.agent.mcts import MCTS
from py_ai.agent.replay_buffer import ReplayBuffer

# Config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1
SIMULATIONS = 25 # Low for speed in dev
GAMES_PER_ITERATION = 2
ITERATIONS = 2 # Short run for test

def self_play(network, card_db):
    game_data = []
    gs = dm_ai_module.GameState(np.random.randint(100000))
    gs.setup_test_duel() # Use test deck for now
    dm_ai_module.PhaseManager.start_game(gs)
    
    mcts = MCTS(network, card_db, simulations=SIMULATIONS)
    
    turn_count = 0
    while True:
        turn_count += 1
        if turn_count > 200: # Safety
            break
            
        # MCTS
        root = mcts.search(gs)
        
        # Policy Target
        policy_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        policy = np.zeros(policy_size)
        
        # Temperature?
        # For now, deterministic or simple proportional
        visits = np.array([child.visit_count for child in root.children])
        actions = [child.action for child in root.children]
        
        if len(visits) == 0:
            break # Game Over
            
        probs = visits / visits.sum()
        
        for child, prob in zip(root.children, probs):
            idx = dm_ai_module.ActionEncoder.action_to_index(child.action)
            if idx >= 0:
                policy[idx] = prob
                
        # Store data
        # State needs to be converted to tensor
        state_tensor = dm_ai_module.TensorConverter.convert_to_tensor(gs, gs.active_player_id)
        # Store (state, policy, value_placeholder, active_player_id)
        game_data.append([state_tensor, policy, None, gs.active_player_id]) 
        
        # Select Action
        # In training, sample from probs. In eval, pick max.
        # Let's sample.
        chosen_child = np.random.choice(root.children, p=probs)
        action = chosen_child.action
        
        # Apply Action
        dm_ai_module.EffectResolver.resolve_action(gs, action, card_db)
        if action.type == dm_ai_module.ActionType.PASS:
            dm_ai_module.PhaseManager.next_phase(gs)
            
        # Check Game Over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(gs)
        if is_over:
            # Result: 1=P1_WIN, 2=P2_WIN, 3=DRAW
            return game_data, result
            
    return game_data, 0 # Draw/Timeout

def train_loop():
    print("Starting Training Loop...")
    
    # Load DB
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cards.csv')
    card_db = dm_ai_module.CsvLoader.load_cards(data_path)
    
    # Init Network
    input_size = dm_ai_module.TensorConverter.INPUT_SIZE
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    network = AlphaZeroNetwork(input_size, action_size)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    
    replay_buffer = ReplayBuffer()
    
    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration + 1}/{ITERATIONS}")
        
        # Self Play
        network.eval()
        for i in range(GAMES_PER_ITERATION):
            print(f"  Self-play game {i+1}...")
            data, result = self_play(network, card_db)
            
            # Assign Values
            # Result: 1=P1 Win (ID 0), 2=P2 Win (ID 1), 3=Draw
            
            for item in data:
                # item: [state, policy, value, player_id]
                player_id = item[3]
                value = 0.0
                
                if result == 1: # P1 Won
                    value = 1.0 if player_id == 0 else -1.0
                elif result == 2: # P2 Won
                    value = 1.0 if player_id == 1 else -1.0
                else: # Draw
                    value = 0.0
                    
                item[2] = value
                
            # Remove player_id before pushing to buffer (Buffer expects 3 items)
            cleaned_data = [item[:3] for item in data]
            replay_buffer.push(cleaned_data)
            
        # Training
        print("  Training...")
        network.train()
        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            if batch:
                states, policies, values = batch
                states = torch.tensor(states, dtype=torch.float32)
                policies = torch.tensor(policies, dtype=torch.float32)
                values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
                
                optimizer.zero_grad()
                pred_policies, pred_values = network(states)
                
                # Loss
                # Value: MSE
                value_loss = ((pred_values - values) ** 2).mean()
                # Policy: Cross Entropy (LogSoftmax + NLL or just -sum(target * log(pred)))
                # pred_policies are logits.
                policy_loss = -(policies * torch.log_softmax(pred_policies, dim=1)).sum(dim=1).mean()
                
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                
                print(f"  Loss: {loss.item():.4f}")
                
        # Save Checkpoint
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f"model_iter_{iteration}.pth")
        torch.save(network.state_dict(), model_path)

if __name__ == "__main__":
    train_loop()
