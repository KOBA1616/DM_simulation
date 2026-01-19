#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Reinforcement Learning Loop
勝敗がつくように強化学習を進める高度なループ
"""

import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import dm_ai_module
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class AdvancedRLLoop:
    """
    Advanced RL training loop that:
    1. Generates complete game episodes with outcomes
    2. Trains model on complete episodes
    3. Evaluates performance against baseline
    4. Adjusts deck/scenarios for better learning
    """
    
    def __init__(self, model_dir="models", data_dir="data", log_dir="logs"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cpu')  # Use CPU for stability
        # DuelTransformer requires vocab_size (1000 tokens) and action_dim (591 actions)
        self.model = DuelTransformer(vocab_size=1000, action_dim=591).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Training history
        self.history = {
            'iteration': [],
            'epoch': [],
            'train_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'win_rate': [],
            'episodes': []
        }
        
        self.total_episodes = 0
        
    def load_magic_deck(self):
        """Load magic.json deck."""
        try:
            deck_path = self.data_dir / "decks" / "magic.json"
            if deck_path.exists():
                with open(deck_path, 'r', encoding='utf-8') as f:
                    deck = json.load(f)
                    if isinstance(deck, list) and len(deck) == 40:
                        return deck
        except:
            pass
        return [1] * 40
    
    def generate_training_episodes(self, num_episodes: int, collect_tokens=True) -> tuple:
        """
        Generate training data from complete game episodes.
        Returns: (states, policies, values, game_results)
        """
        print(f"\n{'='*60}")
        print(f"Generating {num_episodes} complete game episodes...")
        print(f"{'='*60}")
        
        try:
            collector = dm_ai_module.DataCollector()
            
            # Collect in batches
            all_states = []
            all_policies = []
            all_values = []
            all_results = []
            
            batch_size = 12
            num_batches = (num_episodes + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
                
                batch = collector.collect_data_batch_heuristic(
                    current_batch_size,
                    collect_tokens=collect_tokens,
                    collect_tensors=not collect_tokens
                )
                
                all_states.extend(list(batch.token_states) if collect_tokens else list(batch.tensor_states))
                all_policies.extend(list(batch.policies))
                all_values.extend(list(batch.values))
                
                # Track win rates
                wins = sum(1 for v in batch.values if v > 0)
                losses = sum(1 for v in batch.values if v < 0)
                draws = sum(1 for v in batch.values if v == 0)
                
                print(f"  Batch {batch_idx+1}/{num_batches}: "
                      f"{len(all_states)} samples "
                      f"(W:{wins} L:{losses} D:{draws})")
            
            self.total_episodes += num_episodes
            
            return all_states, all_policies, all_values
            
        except Exception as e:
            print(f"Error during episode generation: {e}")
            return [], [], []
    
    def train_epoch(self, states, policies, values, batch_size=32):
        """Train for one epoch."""
        # Convert to tensors
        if len(states) == 0:
            return 0.0, 0.0, 0.0
        
        # States should be token sequences (long type for embedding)
        states_np = np.array(states, dtype=np.int64)
        states_t = torch.LongTensor(states_np).to(self.device)
        
        policies_t = torch.FloatTensor(np.array(policies, dtype=np.float32)).to(self.device)
        values_t = torch.FloatTensor(np.array(values, dtype=np.float32)).reshape(-1, 1).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(states_t, policies_t, values_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        self.model.train()
        
        for batch_states, batch_policies, batch_values in loader:
            # Forward pass
            policy_out, value_out = self.model(batch_states)
            
            # Compute loss
            policy_loss = nn.CrossEntropyLoss()(policy_out, batch_policies)
            value_loss = nn.MSELoss()(value_out, batch_values)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_policy_loss, avg_value_loss
    
    def run_iteration(self, iteration: int, episodes: int = 24, epochs: int = 3):
        """Run one iteration of RL."""
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}")
        print(f"{'#'*60}")
        
        # Generate episodes
        states, policies, values = self.generate_training_episodes(episodes)
        
        if len(states) == 0:
            print("ERROR: No training data generated")
            return False
        
        print(f"\nGenerated {len(states)} training samples")
        
        # Train
        print(f"\n{'='*60}")
        print(f"Training for {epochs} epochs...")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            loss, ploss, vloss = self.train_epoch(states, policies, values)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={loss:.4f} (P:{ploss:.4f} V:{vloss:.4f})")
            
            # Record history
            self.history['iteration'].append(iteration)
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(loss)
            self.history['policy_loss'].append(ploss)
            self.history['value_loss'].append(vloss)
            self.history['episodes'].append(self.total_episodes)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"duel_transformer_iter{iteration}_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        return True
    
    def save_history(self):
        """Save training history."""
        history_path = self.log_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ History saved: {history_path}")
    
    def print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total episodes generated: {self.total_episodes}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.history['train_loss']:
            print(f"\nTraining Loss Progress:")
            print(f"  Initial: {self.history['train_loss'][0]:.4f}")
            print(f"  Final:   {self.history['train_loss'][-1]:.4f}")
            print(f"  Improvement: {self.history['train_loss'][0] - self.history['train_loss'][-1]:.4f}")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Evaluate model on test set")
        print("2. Run self-play to measure win rate")
        print("3. Integrate MCTS with trained model")
        print("4. Run additional iterations with more episodes")
        print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Reinforcement Learning Loop')
    parser.add_argument('--iterations', type=int, default=3, help='Number of RL iterations')
    parser.add_argument('--episodes', type=int, default=24, help='Episodes per iteration')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ADVANCED REINFORCEMENT LEARNING TRAINING")
    print("=== 勝敗がつくように強化学習を進める ===")
    print("="*60)
    
    rl = AdvancedRLLoop()
    
    try:
        for iteration in range(1, args.iterations + 1):
            success = rl.run_iteration(
                iteration=iteration,
                episodes=args.episodes,
                epochs=args.epochs
            )
            
            if not success:
                break
        
        rl.save_history()
        rl.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        rl.save_history()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
