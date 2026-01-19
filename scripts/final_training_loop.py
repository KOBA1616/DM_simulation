#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final RL Training Loop with Generated Data
生成されたデータで強化学習を実行
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
from datetime import datetime
from pathlib import Path

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class FinalTrainingLoop:
    """Final RL training with generated data"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = DuelTransformer(vocab_size=1000, action_dim=591).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        self.history = []
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def load_training_data(self, data_path: str):
        """Load generated training data"""
        print(f"\nLoading training data from {data_path}...")
        
        data = np.load(data_path)
        states = data['states']
        policies = data['policies']
        values = data['values']
        
        print(f"States:   {states.shape}")
        print(f"Policies: {policies.shape}")
        print(f"Values:   {values.shape}")
        
        return states, policies, values
    
    def train(self, states, policies, values, epochs=5, batch_size=32):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"TRAINING FOR {epochs} EPOCHS")
        print(f"{'='*60}\n")
        
        # Convert to tensors
        states_t = torch.LongTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).reshape(-1, 1).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(states_t, policies_t, values_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_ploss = 0.0
            total_vloss = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch_states, batch_policies, batch_values in loader:
                # Forward pass
                policy_out, value_out = self.model(batch_states)
                
                # Loss
                policy_loss = nn.CrossEntropyLoss()(policy_out, batch_policies)
                value_loss = nn.MSELoss()(value_out, batch_values)
                loss = policy_loss + value_loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_ploss = total_ploss / num_batches
            avg_vloss = total_vloss / num_batches
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:7.4f} (P: {avg_ploss:7.4f}, V: {avg_vloss:7.4f})")
            
            self.history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'ploss': avg_ploss,
                'vloss': avg_vloss
            })
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"duel_transformer_final_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        return avg_loss
    
    def evaluate(self, states, values):
        """Simple evaluation on training data"""
        print(f"\n{'='*60}")
        print(f"EVALUATION")
        print(f"{'='*60}\n")
        
        # Check value predictions
        self.model.eval()
        states_t = torch.LongTensor(states[:100]).to(self.device)
        
        with torch.no_grad():
            _, value_predictions = self.model(states_t)
        
        target_values = values[:100]
        
        # Compute accuracy (sign correctness)
        predicted_signs = (value_predictions.cpu().numpy() > 0).astype(int)
        target_signs = (target_values > 0).astype(int)
        
        accuracy = (predicted_signs == target_signs).mean()
        
        print(f"Value prediction accuracy (sign): {accuracy*100:.1f}%")
        print(f"Sample predictions vs targets:")
        for i in range(min(10, len(target_values))):
            pred = value_predictions[i].item()
            target = target_values[i][0]
            print(f"  Sample {i:2d}: Predicted={pred:+.3f}, Target={target:+.1f}, Match={int(pred > 0) == int(target > 0)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Final RL Training')
    parser.add_argument('--data', type=str, default='data/simple_training_data.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FINAL REINFORCEMENT LEARNING TRAINING")
    print("=== 勝敗をつけて強化学習を実行 ===")
    print("="*60)
    
    trainer = FinalTrainingLoop()
    
    # Load data
    states, policies, values = trainer.load_training_data(args.data)
    
    # Train
    final_loss = trainer.train(states, policies, values, epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    trainer.evaluate(states, values)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Training completed")
    print(f"✓ Final loss: {final_loss:.4f}")
    print(f"✓ Model saved to models/")
    print(f"\nNEXT STEPS:")
    print(f"1. Generate more training data with varied scenarios")
    print(f"2. Run additional training iterations")
    print(f"3. Integrate trained model with MCTS")
    print(f"4. Evaluate against baseline AI")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
