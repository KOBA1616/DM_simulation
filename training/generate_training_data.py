#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Training Data Generator
magic.jsonデッキを使用した訓練データ生成スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを設定してパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# カレントディレクトリをプロジェクトルートに変更（データ読み込みのため）
os.chdir(project_root)

import numpy as np
import json
from typing import List

try:
    import dm_ai_module
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def load_magic_deck() -> List[int]:
    """Load the magic.json deck for training."""
    try:
        deck_path = "data/decks/magic.json"
        if os.path.exists(deck_path):
            with open(deck_path, 'r', encoding='utf-8') as f:
                deck = json.load(f)
                if isinstance(deck, list) and len(deck) == 40:
                    print(f"Loaded magic.json deck: {len(deck)} cards")
                    return deck
    except Exception as e:
        print(f"Warning: Failed to load magic.json: {e}")
    
    default = [1] * 40
    print(f"Using default deck: {len(default)} cards (all ID=1)")
    return default

def main():
    print("=" * 80)
    print("TRAINING DATA GENERATOR")
    print("=" * 80)
    
    # Load cards
    print("\n1. Loading card database...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    print("✓ Loaded")
    
    # Load deck
    print("\n2. Loading deck...")
    deck = load_magic_deck()
    
    # Create collector
    print("\n3. Creating DataCollector...")
    collector = dm_ai_module.DataCollector()
    print("✓ Created")
    
    # Collect data
    print("\n4. Collecting training data...")
    print("   Running 12 episodes (target 500+ samples)...")
    
    batch = collector.collect_data_batch_heuristic(12, True, False)
    
    states = list(batch.token_states)
    policies = list(batch.policies)
    values = list(batch.values)
    
    collected = len(states)
    print(f"✓ Collected {collected} samples from first batch")
    
    # Add more if needed
    if collected < 500:
        needed = 500 - collected
        more_episodes = (needed // 40) + 1
        print(f"\n   Collecting {more_episodes} more episodes...")
        more_batch = collector.collect_data_batch_heuristic(more_episodes, True, False)
        
        states.extend(list(more_batch.token_states))
        policies.extend(list(more_batch.policies))
        values.extend(list(more_batch.values))
        
        collected = len(states)
        print(f"✓ Total: {collected} samples")
    
    # Truncate
    states = states[:500]
    policies = policies[:500]
    values = values[:500]
    
    # Convert to numpy
    print("\n5. Converting to numpy arrays...")
    states_np = np.array(states, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.float32)
    values_np = np.array(values, dtype=np.float32).reshape(-1, 1)
    
    print(f"  States:   {states_np.shape}")
    print(f"  Policies: {policies_np.shape}")
    print(f"  Values:   {values_np.shape}")
    
    # Save
    print("\n6. Saving to file...")
    output_path = "data/transformer_training_data.npz"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, states=states_np, policies=policies_np, values=values_np)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved to {output_path}")
    print(f"  File size: {size_mb:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Training data generation complete")
    print("\nNext step:")
    print("  python python/training/train_transformer_phase4.py --epochs 3")
    print("=" * 80)

if __name__ == "__main__":
    main()
