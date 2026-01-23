# Phase 4 Week 2 Day 1 実装計画（1月13日）

**前提条件**:
- ✅ Q1: Synergy初期化 = 手動定義で開始
- ✅ Q2: CLSトークン位置 = 先頭（[CLS] token）
- ✅ Q3: バッチサイズ = 8→16→32→64 段階的拡大
- ✅ Q4: **データ現況 = トレーニングデータなし→新規生成必須**
- ⏳ Q5-Q9: 実装中に決定可能（推奨値あり）

**作業時間配分**: 8時間

---

## Task 1: Synergy マトリックス（手動定義）実装

**所要時間**: 2.5時間  
**担当**: Transformer初期化フェーズ

### 1.1 手動定義ファイル作成

**ファイル作成**: [data/synergy_pairs_v1.json](../../data/synergy_pairs_v1.json)

```json
{
  "description": "Manual Synergy Pairs (Phase 4 v1)",
  "version": "1.0",
  "pairs": [
    {
      "name": "Revolution Change with Multi-Color",
      "card_ids": [101, 205],
      "synergy_score": 0.8,
      "description": "多色カードを踏み台にして革命チェンジする強力なコンボ"
    },
    {
      "name": "Shield Trigger Chain",
      "card_ids": [150, 151],
      "synergy_score": 0.7,
      "description": "シールドトリガーの連鎖効果"
    },
    {
      "name": "Mana Ramp Combo",
      "card_ids": [50, 51, 52],
      "synergy_score": 0.6,
      "description": "マナ加速コンボ（複数カードの相乗効果）"
    },
    {
      "name": "Creature Synergy - Evolution",
      "card_ids": [200, 201, 202],
      "synergy_score": 0.75,
      "description": "進化クリーチャーの進化チェーン"
    }
  ],
  "notes": "カードIDは TOKEN_CARD_OFFSET=100 を基準に設定"
}
```

### 1.2 SynergyGraph への手動定義ロード機能追加

**ファイル修正**: [dm_toolkit/ai/agent/synergy.py](../../dm_toolkit/ai/agent/synergy.py)

```python
# 追加メソッド

@classmethod
def from_manual_pairs(
    cls,
    vocab_size: int,
    pairs_json_path: str,
    embedding_dim: int = 64
) -> 'SynergyGraph':
    """
    手動定義ペアから SynergyGraph を初期化。
    
    Args:
        vocab_size: トークン語彙サイズ（1000）
        pairs_json_path: カード相性ペア JSON ファイルパス
        embedding_dim: シナジー埋め込み次元（64）
    
    Returns:
        SynergyGraph インスタンス
    """
    import json
    
    instance = cls(vocab_size, embedding_dim)
    
    # JSON から相性ペアを読み込み
    with open(pairs_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 固定スコア行列を初期化
    synergy_matrix = torch.zeros(vocab_size, vocab_size)
    
    for pair_info in data['pairs']:
        card_ids = pair_info['card_ids']
        score = pair_info['synergy_score']
        
        # Symmetric な相性スコアを設定
        for i in card_ids:
            for j in card_ids:
                if i != j:
                    synergy_matrix[i, j] = score
    
    # 固定行列をパラメータとして保存（requires_grad=False）
    instance.fixed_synergy_matrix = torch.nn.Parameter(
        synergy_matrix,
        requires_grad=False
    )
    instance.use_fixed_matrix = True
    
    return instance

def get_bias_for_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
    """
    改良版: 固定行列と学習可能埋め込みの両方をサポート
    """
    B, S = sequence.shape
    
    if hasattr(self, 'use_fixed_matrix') and self.use_fixed_matrix:
        # 固定行列を使用
        # sequence[b, s] の値を使用して行列から値を参照
        bias = torch.zeros(B, S, S, device=sequence.device)
        for b in range(B):
            for i in range(S):
                for j in range(S):
                    card_i = sequence[b, i].item()
                    card_j = sequence[b, j].item()
                    if card_i < self.fixed_synergy_matrix.shape[0] and \
                       card_j < self.fixed_synergy_matrix.shape[1]:
                        bias[b, i, j] = self.fixed_synergy_matrix[card_i, card_j]
        return bias
    else:
        # 元の実装: 埋め込みベクトルの内積
        embs = cast(torch.Tensor, self.synergy_embeddings(sequence))
        bias = torch.bmm(embs, embs.transpose(1, 2))
        bias = bias / (self.embedding_dim ** 0.5)
        return bias
```

### 1.3 単体テスト実装

**ファイル作成**: [tests/test_synergy_manual.py](../../tests/test_synergy_manual.py)

```python
import pytest
import torch
import json
import tempfile
import os
from dm_toolkit.ai.agent.synergy import SynergyGraph

def test_synergy_from_manual_pairs():
    """Test loading synergy from manual pairs JSON."""
    
    # テスト用 JSON を一時作成
    pairs_data = {
        "pairs": [
            {"card_ids": [100, 101], "synergy_score": 0.8},
            {"card_ids": [200, 201], "synergy_score": 0.6},
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pairs_data, f)
        temp_path = f.name
    
    try:
        # SynergyGraph を作成
        synergy = SynergyGraph.from_manual_pairs(
            vocab_size=1000,
            pairs_json_path=temp_path,
            embedding_dim=64
        )
        
        # 固定行列が設定されているか確認
        assert hasattr(synergy, 'fixed_synergy_matrix')
        assert synergy.use_fixed_matrix
        
        # スコアが正確に設定されているか確認
        assert synergy.fixed_synergy_matrix[100, 101].item() == 0.8
        assert synergy.fixed_synergy_matrix[200, 201].item() == 0.6
        
        # Symmetric 性確認（オプション）
        assert synergy.fixed_synergy_matrix[101, 100].item() == 0.8
        
        # get_bias_for_sequence() で値が取得できるか確認
        tokens = torch.tensor([[100, 101, 0], [200, 201, 0]])  # [batch=2, seq=3]
        bias = synergy.get_bias_for_sequence(tokens)
        
        assert bias.shape == (2, 3, 3)
        assert bias[0, 0, 1].item() == 0.8  # card 100-101 相性
        
        print("✅ test_synergy_from_manual_pairs passed")
        
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    test_synergy_from_manual_pairs()
```

**実行コマンド**:
```bash
pytest tests/test_synergy_manual.py -v
```

**チェックリスト**:
- [ ] synergy_pairs_v1.json 作成
- [ ] SynergyGraph.from_manual_pairs() 実装
- [ ] get_bias_for_sequence() 改良
- [ ] test_synergy_manual.py ✅ 実行成功

---

## Task 2: トレーニングデータ生成パイプライン

**所要時間**: 3.0時間  
**担当**: データ準備フェーズ

### 2.1 データ生成スクリプト作成

**ファイル作成**: [generate_transformer_training_data.py](../../generate_transformer_training_data.py)

```python
#!/usr/bin/env python3
"""
Generate Transformer training data from self-play scenarios.

Output format:
    - tokens: [num_samples, seq_len] int64 token IDs
    - policies: [num_samples, action_dim] float32 policy targets
    - values: [num_samples, 1] float32 value targets
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import List, Tuple
from tqdm import tqdm

# Setup paths
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Dynamic import
dm_ai_module = None
try:
    import dm_ai_module
except ImportError:
    print("⚠️  Could not import dm_ai_module. Token generation will be mocked.")

from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.synergy import SynergyGraph
from dm_toolkit.training.scenario_runner import ScenarioRunner

def generate_samples(
    num_samples: int = 1000,
    output_path: str = "data/training_data.npz",
    vocab_size: int = 1000,
    max_seq_len: int = 200
) -> None:
    """Generate Transformer training data from scenarios."""
    
    print(f"Generating {num_samples} training samples...")
    
    all_tokens = []
    all_policies = []
    all_values = []
    
    # Scenario data を読み込み
    runner = ScenarioRunner(scenario_names=['basic', 'advanced'])
    
    for sample_idx in tqdm(range(num_samples)):
        try:
            # ゲーム 1 試行を実行
            game_data = runner.run_scenario()
            
            # GameState → Tokens（C++ TensorConverter 使用）
            if dm_ai_module:
                tokens = dm_ai_module.convert_to_sequence(
                    game_data['state'],
                    player_view=0,
                    mask_opponent_hand=True
                )
            else:
                # Fallback: ランダムトークン（テスト用）
                tokens = np.random.randint(
                    0, vocab_size,
                    size=np.random.randint(50, max_seq_len)
                )
            
            # パディング
            if len(tokens) < max_seq_len:
                tokens = np.pad(
                    tokens,
                    (0, max_seq_len - len(tokens)),
                    constant_values=0
                )
            else:
                tokens = tokens[:max_seq_len]
            
            all_tokens.append(tokens)
            
            # Policy & Value targets
            policy_target = game_data['policy']  # [action_dim]
            value_target = game_data['value']     # scalar
            
            all_policies.append(policy_target)
            all_values.append([value_target])
        
        except Exception as e:
            print(f"  ⚠️  Sample {sample_idx} generation failed: {e}")
            continue
    
    # NPZ で保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tokens_array = np.array(all_tokens, dtype=np.int64)
    policies_array = np.array(all_policies, dtype=np.float32)
    values_array = np.array(all_values, dtype=np.float32)
    
    np.savez(
        output_path,
        tokens=tokens_array,
        policies=policies_array,
        values=values_array
    )
    
    print(f"\n✅ Generated {len(all_tokens)} samples")
    print(f"   Tokens shape: {tokens_array.shape}")
    print(f"   Policies shape: {policies_array.shape}")
    print(f"   Values shape: {values_array.shape}")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Transformer training data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/training_data.npz", help="Output path")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=200, help="Max sequence length")
    
    args = parser.parse_args()
    
    generate_samples(
        num_samples=args.num_samples,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )
```

**実行例**:
```bash
python generate_transformer_training_data.py --num-samples 1000 --output data/training_data.npz
```

### 2.2 データロード検証スクリプト

**ファイル作成**: [tests/test_training_data_load.py](../../tests/test_training_data_load.py)

```python
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from dm_toolkit.training.training_pipeline import DuelDataset, collate_batch

def test_training_data_load_and_batch():
    """Test loading generated training data and batching."""
    
    # テスト用データ生成
    num_samples = 10
    max_seq_len = 200
    action_dim = 100
    
    tokens = np.random.randint(0, 1000, size=(num_samples, max_seq_len), dtype=np.int64)
    policies = np.random.randn(num_samples, action_dim).astype(np.float32)
    values = np.random.randn(num_samples, 1).astype(np.float32)
    
    # Dataset 作成
    tokens_list = [torch.from_numpy(tokens[i]) for i in range(num_samples)]
    dataset = DuelDataset(
        tokens=tokens_list,
        states=None,
        policies=torch.from_numpy(policies),
        values=torch.from_numpy(values)
    )
    
    # DataLoader でバッチ処理
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_batch
    )
    
    # バッチを確認
    for batch_idx, batch in enumerate(loader):
        assert 'tokens' in batch
        assert 'padding_mask' in batch
        assert 'policy' in batch
        assert 'value' in batch
        
        tokens_batch = batch['tokens']
        padding_mask = batch['padding_mask']
        
        assert tokens_batch.shape == (4, max_seq_len)
        assert padding_mask.shape == (4, max_seq_len)
        assert tokens_batch.dtype == torch.int64
        assert padding_mask.dtype == torch.bool
        
        print(f"✅ Batch {batch_idx}: tokens {tokens_batch.shape}, mask {padding_mask.shape}")
    
    print("✅ test_training_data_load_and_batch passed")

if __name__ == "__main__":
    test_training_data_load_and_batch()
```

**チェックリスト**:
- [ ] generate_transformer_training_data.py 実装
- [ ] 最初の 100 サンプル生成・確認
- [ ] test_training_data_load.py ✅ 実行成功
- [ ] tokens shape が [N, 200] で統一されているか確認

---

## Task 3: Transformer 学習スクリプト実装

**所要時間**: 2.5時間  
**担当**: モデル統合テストフェーズ

### 3.1 基本学習ループ実装

**ファイル作成**: [train_transformer_phase4.py](../../train_transformer_phase4.py)

```python
#!/usr/bin/env python3
"""
Phase 4 Transformer Training Script

Week 2 Goal: Verify model initialization and basic training loop
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.synergy import SynergyGraph
from dm_toolkit.training.training_pipeline import DuelDataset, collate_batch

class TransformerTrainer:
    def __init__(
        self,
        vocab_size: int = 1000,
        action_dim: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_len: int = 200,
        synergy_pairs_path: str = "data/synergy_pairs_v1.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        
        print(f"🔧 Initializing TransformerTrainer on {device.upper()}")
        
        # 1. Synergy Graph ロード
        try:
            self.synergy_graph = SynergyGraph.from_manual_pairs(
                vocab_size=vocab_size,
                pairs_json_path=synergy_pairs_path
            )
            print(f"✅ Synergy pairs loaded from {synergy_pairs_path}")
        except Exception as e:
            print(f"⚠️  Could not load synergy pairs: {e}")
            print("   Using default (random) synergy initialization")
            self.synergy_graph = SynergyGraph(vocab_size=vocab_size)
        
        # 2. Model 初期化
        self.model = DuelTransformer(
            vocab_size=vocab_size,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len,
            synergy_matrix_path=None  # SynergyGraph で管理
        ).to(device)
        
        # Synergy Graph を model に結合（共有）
        self.model.synergy_graph = self.synergy_graph
        
        print(f"✅ Model initialized")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - Device: {device}")
        
        # 3. Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 4. Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Metrics
        self.metrics = {
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_total_loss': [],
            'val_policy_loss': [],
            'val_value_loss': []
        }
    
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        policy_losses = []
        value_losses = []
        total_losses = []
        
        for batch_idx, batch in enumerate(loader):
            tokens = batch['tokens'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            policy_target = batch['policy'].to(self.device)
            value_target = batch['value'].to(self.device)
            
            # Forward pass
            policy_logits, value_pred = self.model(tokens, padding_mask)
            
            # Loss computation
            policy_loss = F.cross_entropy(
                policy_logits,
                policy_target.argmax(dim=1)
            )
            value_loss = F.mse_loss(value_pred, value_target)
            total_loss = policy_loss + value_loss
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            total_losses.append(total_loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}] "
                      f"Policy Loss: {policy_loss.item():.4f}, "
                      f"Value Loss: {value_loss.item():.4f}")
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'total_loss': np.mean(total_losses)
        }
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Validate on a dataset."""
        self.model.eval()
        
        policy_losses = []
        value_losses = []
        
        for batch in loader:
            tokens = batch['tokens'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            policy_target = batch['policy'].to(self.device)
            value_target = batch['value'].to(self.device)
            
            policy_logits, value_pred = self.model(tokens, padding_mask)
            
            policy_loss = F.cross_entropy(
                policy_logits,
                policy_target.argmax(dim=1)
            )
            value_loss = F.mse_loss(value_pred, value_target)
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        checkpoint_dir: str = "checkpoints/phase4"
    ) -> None:
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n🚀 Starting training: {epochs} epochs")
        print(f"   Checkpoint dir: {checkpoint_dir}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"  Train - Policy Loss: {train_metrics['policy_loss']:.4f}, "
                  f"Value Loss: {train_metrics['value_loss']:.4f}")
            
            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                total_val_loss = val_metrics['policy_loss'] + val_metrics['value_loss']
                print(f"  Val   - Policy Loss: {val_metrics['policy_loss']:.4f}, "
                      f"Value Loss: {val_metrics['value_loss']:.4f}")
                
                # Save best checkpoint
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    self._save_checkpoint(
                        epoch,
                        checkpoint_dir,
                        train_metrics,
                        val_metrics
                    )
            
            # LR schedule
            self.scheduler.step()
            
            print()
    
    def _save_checkpoint(self, epoch, checkpoint_dir, train_metrics, val_metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:02d}.pt")
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Transformer Training")
    parser.add_argument("--data", type=str, default="data/training_data.npz", help="Training data path")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--synergy-pairs", type=str, default="data/synergy_pairs_v1.json", help="Synergy pairs file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase4", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.data):
        print(f"❌ Training data not found: {args.data}")
        print("   Run: python generate_transformer_training_data.py")
        return
    
    print(f"📂 Loading data from {args.data}...")
    data = np.load(args.data)
    
    tokens_list = [torch.from_numpy(data['tokens'][i]) for i in range(len(data['tokens']))]
    policies = torch.from_numpy(data['policies'])
    values = torch.from_numpy(data['values'])
    
    # Train/Val split
    num_train = int(0.8 * len(tokens_list))
    
    train_tokens = tokens_list[:num_train]
    train_policies = policies[:num_train]
    train_values = values[:num_train]
    
    val_tokens = tokens_list[num_train:]
    val_policies = policies[num_train:]
    val_values = values[num_train:]
    
    train_dataset = DuelDataset(
        tokens=train_tokens,
        states=None,
        policies=train_policies,
        values=train_values
    )
    
    val_dataset = DuelDataset(
        tokens=val_tokens,
        states=None,
        policies=val_policies,
        values=val_values
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch
    )
    
    # Train
    trainer = TransformerTrainer(synergy_pairs_path=args.synergy_pairs)
    trainer.train(train_loader, val_loader, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir)

if __name__ == "__main__":
    main()
```

**実行例**:
```bash
# Step 1: データ生成
python generate_transformer_training_data.py --num-samples 1000

# Step 2: 学習開始（バッチサイズ 8）
python train_transformer_phase4.py --batch-size 8 --epochs 10

# Step 3: バッチサイズ段階的拡大
python train_transformer_phase4.py --batch-size 16 --epochs 5
python train_transformer_phase4.py --batch-size 32 --epochs 5
python train_transformer_phase4.py --batch-size 64 --epochs 5
```

**チェックリスト**:
- [ ] train_transformer_phase4.py 実装
- [ ] TransformerTrainer クラス ✅ 動作確認
- [ ] バッチサイズ 8 で 1 epoch ✅ 完了
- [ ] Loss 曲線が低下していることを確認

---

## Task 4: ハイパーパラメータ検証と記録

**所要時間**: 0.5時間  
**担当**: モニタリングフェーズ

### 4.1 バッチサイズ段階的拡大テスト

**スクリプト**: [test_batch_scaling.py](../../test_batch_scaling.py)

```bash
# 各バッチサイズでメモリ使用量と速度を測定
for batch_size in 8 16 32 64; do
    echo "Testing batch size $batch_size..."
    python train_transformer_phase4.py \
        --batch-size $batch_size \
        --epochs 1 \
        --checkpoint-dir "checkpoints/batch_test_$batch_size"
done
```

**期待される結果**:
- バッチサイズ 8: メモリ ~2GB, 速度 ~50 samples/sec
- バッチサイズ 16: メモリ ~3.5GB, 速度 ~80 samples/sec
- バッチサイズ 32: メモリ ~6GB, 速度 ~120 samples/sec
- バッチサイズ 64: メモリ ~10GB (RTX 3090 では OOM の可能性)

**チェックリスト**:
- [ ] 各バッチサイズで正常動作確認
- [ ] メモリ使用量を記録
- [ ] 最適バッチサイズを決定（推奨: 32）

---

## 実装スケジュール

```
2026年1月13日（Week 2 Day 1）

10:00-12:30 : Task 1 (2.5h)
  ✓ synergy_pairs_v1.json 作成
  ✓ SynergyGraph.from_manual_pairs() 実装
  ✓ test_synergy_manual.py ✅

12:30-13:00 : 昼食

13:00-16:00 : Task 2 (3.0h)
  ✓ generate_transformer_training_data.py 実装
  ✓ 初期 100 サンプル生成
  ✓ test_training_data_load.py ✅

16:00-18:30 : Task 3 (2.5h)
  ✓ train_transformer_phase4.py 実装
  ✓ TransformerTrainer クラス
  ✓ バッチサイズ 8 で 1 epoch ✅

18:30-19:00 : Task 4 (0.5h)
  ✓ バッチサイズスケーリング検証
  ✓ ログ記録

合計: 8時間
```

---

## Q5-Q9 の決定基準（実装中に確認）

| 質問 | Week 2 Day 1 での決定タイミング | 推奨値 |
|------|------|------|
| Q5: Pos Encoding | Model initialization 時 | 学習可能（現行） |
| Q6: データ拡張 | Dataset 実装時 | Phase 2 延期 |
| Q7: 評価指標 | Trainer metrics 実装時 | vs Random + vs MLP |
| Q8: デプロイ基準 | Validation metrics 確定時 | vs MLP ≥ 55% |
| Q9: Synergy Matrix | Task 1 で確定 | 密行列OK（4MB） |

---

## Week 2 Day 1 の最終チェックリスト

### Phase 1: セットアップ
- [ ] GPU 環境確認（cuda_available=True）
- [ ] メモリ確認（≥ 12GB 推奨）
- [ ] PyTorch 2.0+ インストール確認

### Phase 2: データ準備
- [ ] synergy_pairs_v1.json 作成 ✅
- [ ] SynergyGraph 実装 ✅
- [ ] Training data 1000 サンプル生成 ✅

### Phase 3: モデル・学習
- [ ] DuelTransformer 初期化 ✅
- [ ] TransformerTrainer クラス ✅
- [ ] 学習ループ 1 epoch 完了 ✅

### Phase 4: 検証
- [ ] Loss 曲線グラフ作成
- [ ] バッチサイズ段階的拡大テスト ✅
- [ ] メモリ使用量記録

### 最終確認
- [ ] すべてのテスト ✅ 通過
- [ ] チェックポイント保存確認
- [ ] ログファイル生成

**Week 2 Day 1 完了時の予期される成果**:
✅ DuelTransformer が通常に訓練できることを実証  
✅ Policy Loss と Value Loss が低下するトレンドを確認  
✅ バッチサイズ最適値（推奨 32）を決定  
✅ Week 2 Day 2-3 への引き継ぎ準備完了
