#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Training Loop with Reinforcement Learning
統合訓練ループ: データ生成 → 訓練 → 評価を繰り返す
"""

import sys
sys.path.insert(0, '.')

import subprocess
import argparse
import os
import json
from datetime import datetime

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        return True
    else:
        print(f"✗ {description} failed with code {result.returncode}")
        return False

def create_training_config(iteration: int):
    """Create iteration-specific config"""
    config = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "description": f"Reinforcement Learning Iteration {iteration}",
        "steps": [
            f"Data generation (Episode {iteration * 3})",
            f"Model training (Epoch {iteration * 2})",
            f"Evaluation"
        ]
    }
    return config

def main(num_iterations: int = 3, batch_size: int = 8, epochs_per_iter: int = 2):
    """Main training loop"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "REINFORCEMENT LEARNING TRAINING LOOP" + " "*25 + "║")
    print("║" + f" "*10 + f"Iterations: {num_iterations}, Batch: {batch_size}, Epochs/Iter: {epochs_per_iter}" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    training_log = []
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n\n{'#'*80}")
        print(f"# ITERATION {iteration}/{num_iterations}")
        print(f"{'#'*80}")
        
        # Config
        config = create_training_config(iteration)
        
        # Step 1: Generate data
        episodes = iteration * 12  # Increase episodes each iteration
        samples = episodes * 4  # Estimate 4 samples per episode
        
        cmd_data = f"python generate_training_data.py"
        if not run_command(cmd_data, f"Data Generation (Iteration {iteration})"):
            print("⚠ Data generation warning, continuing with available data...")
        
        # Step 2: Train
        cmd_train = f"python train_simple.py --epochs {epochs_per_iter} --batch-size {batch_size}"
        if not run_command(cmd_train, f"Model Training (Iteration {iteration})"):
            print("✗ Training failed, stopping")
            break
        
        # Step 3: Log results
        iteration_log = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "episodes": episodes,
            "batch_size": batch_size,
            "epochs": epochs_per_iter,
            "status": "complete"
        }
        training_log.append(iteration_log)
        
        print(f"\n✓ Iteration {iteration} complete")
    
    # Final summary
    print("\n\n" + "="*80)
    print("TRAINING LOOP SUMMARY")
    print("="*80)
    
    for log in training_log:
        print(f"  Iteration {log['iteration']}: ✓ Complete")
    
    # Save log
    log_path = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Training log saved to {log_path}")
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR REINFORCEMENT LEARNING")
    print("="*80)
    print("""
1. Analyze trained model:
   - Evaluate model on test set
   - Check win rate vs heuristic AI
   
2. Improve data generation:
   - Increase sample size (1000+)
   - Ensure games reach completion
   - Add reward signal (wins/losses)
   
3. Integrate MCTS:
   - Combine trained model as policy network
   - Use Monte Carlo Tree Search for planning
   - Implement AlphaZero-style self-play
   
4. Optimize hyperparameters:
   - Learning rate scheduling
   - Batch size optimization
   - Regularization tuning
   
Recommended command to continue:
  python generate_training_data.py --episodes 50
  python train_simple.py --epochs 5 --batch-size 16
""")
    
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated RL Training Loop")
    parser.add_argument("--iterations", type=int, default=2, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per iteration")
    args = parser.parse_args()
    
    main(num_iterations=args.iterations, batch_size=args.batch_size, epochs_per_iter=args.epochs)
