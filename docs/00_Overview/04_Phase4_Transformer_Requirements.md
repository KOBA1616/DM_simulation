# Phase 4: Transformer実装 詳細要件定義書

**文書ID**: REQ-PHASE4-TRANSFORMER-001  
**作成日**: 2026年1月9日  
**ステータス**: Draft  
**優先度**: High  
**想定期間**: 2週間（Week 2-3）

---

## 1. 概要と目的

### 1.1 背景
現行のMLPベースAIモデルは固定長特徴ベクトル（INPUT_SIZE=143）を使用しており、以下の制約がある：
- カード間の関係性を明示的にモデル化できない
- 可変長の状態（手札、マナ、バトルゾーン）を適切に表現できない
- シーケンシャルな情報（カードのプレイ順等）を失う

### 1.2 目的
Transformerアーキテクチャを導入し、以下を実現する：
- **シーケンス理解**: カード間の依存関係と相互作用を学習
- **Synergy認識**: カードの相性（コンボ）を明示的にモデル化
- **スケーラビリティ**: より大規模なカードプールへの対応
- **性能向上**: MLPモデル比で勝率5-10%向上を目標

### 1.3 成功基準
- [ ] Transformerモデルの学習パイプライン稼働
- [ ] MLPと同等以上の性能（vs Random: 85%+）
- [ ] 推論速度 < 10ms/action
- [ ] 24時間連続学習で安定動作
- [ ] メモリ効率: VRAM使用量 < 8GB（バッチサイズ64）

---

## 2. アーキテクチャ詳細仕様

### 2.1 モデル構造

#### 2.1.1 全体構成
```
[入力層] トークンシーケンス (Batch, SeqLen)
    ↓
[Embedding層] Token + Positional Embeddings
    ↓
[Synergy Bias] カード相性マトリクス注入
    ↓
[Transformer Encoder] × 6層
    ↓
[Pooling] CLS Token抽出
    ↓
[Policy Head] 行動確率分布 (action_dim=500)
    ↓
[Value Head] 局面評価値 (scalar)
```

#### 2.1.2 ハイパーパラメータ
| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| d_model | 256 | 計算コストとモデル容量のバランス |
| nhead | 8 | d_model=256 → 各head=32次元 |
| num_layers | 6 | GPT-2 Small相当（過学習防止） |
| dim_feedforward | 1024 | 標準的な4×d_model |
| max_seq_len | 200 | 最大ゲーム状態を収容 |
| vocab_size | 1000 | カードID + 特殊トークン |
| dropout | 0.1 | 正則化 |
| activation | GELU | Transformer標準 |

#### 2.1.3 入力仕様

**トークン構造**:
```
[CLS] [GLOBAL_INFO] [SEP] 
[SELF_HAND_START] card1 card2 ... [SEP]
[SELF_MANA_START] card1 card2 ... [SEP]
[SELF_BATTLE_START] card1 card2 ... [SEP]
[SELF_GRAVE_START] card1 card2 ... [SEP]
[SELF_SHIELD_START] shield1 shield2 ... [SEP]
[OPP_HAND_START] <masked> ... [SEP]
[OPP_MANA_START] card1 card2 ... [SEP]
...
[PAD] [PAD] ... (最大200まで)
```

**特殊トークン定義**:
```cpp
TOKEN_PAD = 0           // パディング
TOKEN_CLS = 1           // 分類トークン（文頭）
TOKEN_SEP = 2           // セパレータ
TOKEN_SELF_HAND_START = 3
TOKEN_SELF_MANA_START = 4
TOKEN_SELF_BATTLE_START = 5
TOKEN_SELF_GRAVE_START = 6
TOKEN_SELF_SHIELD_START = 7
TOKEN_OPP_HAND_START = 8
TOKEN_OPP_MANA_START = 9
TOKEN_OPP_BATTLE_START = 10
TOKEN_OPP_GRAVE_START = 11
TOKEN_OPP_SHIELD_START = 12
TOKEN_GLOBAL_START = 13
TOKEN_CARD_OFFSET = 100  // カードID開始位置
```

**グローバル情報エンコーディング**:
- ターン数: 直接埋め込み（0-20ターン想定）
- フェーズ: ワンホット（7フェーズ）
- マナ数（自分/相手）: 値の埋め込み
- シールド数（自分/相手）: 値の埋め込み

#### 2.1.4 Synergy Bias Matrix
カード間の相性を事前学習または手動定義で構築：

```python
synergy_matrix[card_i, card_j] = synergy_score
# synergy_score ∈ [-1.0, 1.0]
#   1.0: 強いコンボ（例: 革命チェンジペア）
#   0.0: 無関係
#  -1.0: アンチシナジー
```

**初期実装戦略**:
1. 手動定義（既知のコンボカードペア）
2. 対戦データから共起頻度で推定
3. 将来的にGraph Neural Networkで学習

---

### 2.2 C++ TensorConverter拡張

#### 2.2.1 既存実装の確認
現在実装済み：
- `convert_to_sequence()`: GameState → トークンシーケンス
- `convert_batch_sequence()`: バッチ処理版
- 定数: `MAX_SEQ_LEN=200`, `VOCAB_SIZE=1000`

#### 2.2.2 必要な拡張
1. **パディング処理の最適化**
   ```cpp
   // 現状: ゼロパディング
   // 改善: attention_maskの生成
   std::pair<std::vector<long>, std::vector<bool>> 
   convert_to_sequence_with_mask(const GameState& state);
   ```

2. **バッチ最適化**
   - メモリプールの導入（再確保を削減）
   - SIMDによる高速化検討

3. **デバッグ情報**
   ```cpp
   struct SequenceDebugInfo {
       int total_tokens;
       int padding_tokens;
       std::map<std::string, int> zone_sizes;
   };
   ```

---

### 2.3 学習パイプライン設計

#### 2.3.1 データフロー
```
[C++ 対戦実行] 
    ↓ convert_batch_sequence()
[トークンシーケンス] (.npy)
    ↓ TransformerDataset
[DataLoader] (パディング処理)
    ↓
[TransformerTrainer]
    ↓
[Loss計算] Policy + Value
    ↓
[Optimizer] Adam (lr=1e-4)
    ↓
[Checkpoint保存]
```

#### 2.3.2 損失関数
```python
class AlphaZeroLoss(nn.Module):
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        # Policy: Cross-Entropy
        policy_loss = F.cross_entropy(policy_pred, policy_target)
        
        # Value: MSE
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_target)
        
        # 合計（重み付け）
        total_loss = policy_loss + 0.5 * value_loss
        return total_loss, policy_loss, value_loss
```

#### 2.3.3 最適化戦略
- **Optimizer**: Adam (β1=0.9, β2=0.999, eps=1e-8)
- **学習率スケジュール**: Cosine Annealing with Warmup
  - Warmup: 1000ステップで0→1e-4
  - Cosine: 1e-4 → 1e-6（50エポック）
- **Gradient Clipping**: max_norm=1.0
- **Weight Decay**: 0.01（AdamW）

---

## 3. 実装タスク詳細

### 3.1 Week 2: アーキテクチャと学習ループ

#### Day 1: データパイプライン設計
**成果物**: データ仕様書、サンプルデータ生成スクリプト

**タスク**:
1. `TransformerDataset`クラスの実装
   - [ ] `.npy`ファイルからトークンシーケンス読み込み
   - [ ] Policy/Valueターゲットの読み込み
   - [ ] Vocab sizeの自動推定

2. `collate_fn`のカスタマイズ
   - [ ] 可変長シーケンスのパディング
   - [ ] attention_mask生成
   - [ ] バッチテンソルの構築

3. サンプルデータ生成
   ```bash
   # C++側でデータ生成
   python scripts/generate_training_data.py \
     --mode transformer \
     --games 1000 \
     --output data/transformer_train.npy
   ```

**検証**:
```python
# データローダーのテスト
dataset = TransformerDataset("data/transformer_train.npy")
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
batch = next(iter(loader))
print(f"Sequences: {batch[0].shape}")  # [16, max_len]
print(f"Policies: {batch[1].shape}")   # [16, action_dim]
print(f"Values: {batch[2].shape}")     # [16]
```

---

#### Day 2: モデル構造の詳細設計

**成果物**: 設計ドキュメント、モデル実装レビュー

**タスク**:
1. `DuelTransformer`の検証
   - [ ] 既存実装のレビュー
   - [ ] forward()メソッドの動作確認
   - [ ] Synergy Biasの適用検証

2. Synergy Matrix初期実装
   - [ ] 手動定義ファイル（JSON/CSV）
   - [ ] ロード機能の実装
   - [ ] デフォルト値（全て0.0）のフォールバック

3. ユニットテスト
   ```python
   def test_transformer_forward():
       model = DuelTransformer(vocab_size=1000, action_dim=500)
       x = torch.randint(0, 1000, (4, 100))  # Batch=4, Seq=100
       policy, value = model(x)
       assert policy.shape == (4, 500)
       assert value.shape == (4, 1)
   ```

**疑問点（要確認）**:
- [ ] Synergy Matrixのサイズ: 1000×1000は大きすぎる？→ 疎行列化検討
- [ ] CLSトークンの最適位置: 先頭 vs 末尾？
- [ ] Positional Encodingは学習可能 or 固定？ → 現状: 学習可能

---

#### Day 3: Trainerクラス実装

**成果物**: `train_transformer.py`（基本バージョン）

**実装**:
```python
class TransformerTrainer:
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.loss_fn = AlphaZeroLoss()
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            seqs, policies, values = batch
            seqs = seqs.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            
            # Forward
            pred_policy, pred_value = self.model(seqs)
            
            # Loss
            loss, p_loss, v_loss = self.loss_fn(
                pred_policy, pred_value, policies, values
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def validate(self, dataloader):
        # 省略（実装必要）
        pass
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
```

**タスク**:
- [ ] train_epoch()の実装
- [ ] validate()の実装
- [ ] チェックポイント保存/読み込み
- [ ] ロギング（TensorBoard連携）

---

#### Day 4: データローダーとバッチ処理

**タスク**:
1. 効率的なデータ読み込み
   - [ ] メモリマップモード（大規模データ対応）
   - [ ] プリフェッチング
   - [ ] マルチプロセスローディング

2. データ拡張（オプション）
   - [ ] カード順のランダムシャッフル（同一ゾーン内）
   - [ ] トークンドロップアウト（0.05確率で<UNK>に置換）

3. バリデーションセット分割
   ```python
   from sklearn.model_selection import train_test_split
   
   train_idx, val_idx = train_test_split(
       range(len(dataset)), test_size=0.1, random_state=42
   )
   train_subset = Subset(dataset, train_idx)
   val_subset = Subset(dataset, val_idx)
   ```

---

#### Day 5: ロギングとチェックポイント管理

**タスク**:
1. TensorBoard統合
   ```python
   from torch.utils.tensorboard import SummaryWriter
   
   writer = SummaryWriter('runs/transformer_v1')
   writer.add_scalar('Loss/train', train_loss, epoch)
   writer.add_scalar('Loss/val', val_loss, epoch)
   writer.add_scalar('Learning_Rate', lr, epoch)
   ```

2. チェックポイント戦略
   - 毎エポック保存: `checkpoint_epoch_{epoch}.pt`
   - Best model保存: `best_model.pt`（val_loss基準）
   - 最新3つのみ保持（ディスク節約）

3. Early Stopping実装
   ```python
   class EarlyStopping:
       def __init__(self, patience=5, min_delta=1e-4):
           self.patience = patience
           self.min_delta = min_delta
           self.counter = 0
           self.best_loss = None
       
       def __call__(self, val_loss):
           if self.best_loss is None:
               self.best_loss = val_loss
           elif val_loss > self.best_loss - self.min_delta:
               self.counter += 1
               if self.counter >= self.patience:
                   return True  # Stop training
           else:
               self.best_loss = val_loss
               self.counter = 0
           return False
   ```

---

### 3.2 Week 3: TensorConverter連携とベンチマーク

#### Day 1: C++バインディング強化

**タスク**:
1. `convert_batch_sequence_with_mask()`の実装
   ```cpp
   struct SequenceBatch {
       std::vector<long> tokens;  // Flattened [Batch*SeqLen]
       std::vector<bool> mask;    // Flattened [Batch*SeqLen]
       int batch_size;
       int seq_len;
   };
   
   SequenceBatch convert_batch_sequence_with_mask(...);
   ```

2. Pythonバインディング更新
   ```cpp
   // bind_ai.cpp
   py::class_<SequenceBatch>(m, "SequenceBatch")
       .def_readonly("tokens", &SequenceBatch::tokens)
       .def_readonly("mask", &SequenceBatch::mask)
       .def_readonly("batch_size", &SequenceBatch::batch_size)
       .def_readonly("seq_len", &SequenceBatch::seq_len);
   ```

3. Python側の使用例
   ```python
   import dm_ai_module as dm
   
   batch = dm.TensorConverter.convert_batch_sequence_with_mask(
       states, card_db, mask_opponent_hand=True
   )
   tokens = torch.tensor(batch.tokens).view(batch.batch_size, batch.seq_len)
   mask = torch.tensor(batch.mask).view(batch.batch_size, batch.seq_len)
   ```

---

#### Day 2: パフォーマンス最適化

**最適化項目**:
1. **メモリコピー削減**
   - Zero-copy interfaceの検討（PyTorch C++ extension）
   - Buffer再利用（プーリング）

2. **並列化**
   - OpenMPによるバッチ並列変換
   - CPUスレッド数の調整

3. **プロファイリング**
   ```python
   import cProfile
   
   profiler = cProfile.Profile()
   profiler.enable()
   # 変換処理
   batch = convert_batch_sequence(states, card_db)
   profiler.disable()
   profiler.print_stats(sort='cumtime')
   ```

**目標**:
- 変換オーバーヘッド: < 5ms/batch (64サンプル)
- メモリ効率: > 90%（無駄なコピー < 10%）

---

#### Day 3-5: 統合テストとベンチマーク

**テスト項目**:

1. **精度検証**
   ```python
   # MLPとの比較実験
   results = {
       'MLP': {'win_rate': 0.83, 'avg_turns': 8.2},
       'Transformer': {'win_rate': 0.??, 'avg_turns': ??}
   }
   ```

2. **速度ベンチマーク**
   ```python
   import time
   
   # 推論速度測定
   model.eval()
   with torch.no_grad():
       start = time.time()
       for _ in range(1000):
           policy, value = model(sample_input)
       end = time.time()
   
   avg_time_ms = (end - start) / 1000 * 1000
   print(f"Average inference time: {avg_time_ms:.2f}ms")
   ```

3. **メモリ測定**
   ```python
   import torch.cuda as cuda
   
   cuda.reset_peak_memory_stats()
   # 学習1イテレーション
   loss.backward()
   optimizer.step()
   
   max_memory = cuda.max_memory_allocated() / 1024**3  # GB
   print(f"Peak GPU memory: {max_memory:.2f}GB")
   ```

4. **長時間安定性テスト**
   - 24時間連続学習
   - メモリリークチェック
   - Loss発散の監視

**ベンチマーク環境**:
- GPU: NVIDIA RTX 3090
- CUDA: 11.8
- PyTorch: 2.0+
- バッチサイズ: 64
- シーケンス長: 平均150、最大200

**合格基準**:
- [ ] vs Random勝率 ≥ 85%
- [ ] vs MLP勝率 ≥ 50%（同等以上）
- [ ] 推論速度 < 10ms/action
- [ ] VRAM使用量 < 8GB
- [ ] 24時間稼働で異常なし

---

## 4. データ要件

### 4.1 学習データ
- **サンプル数**: 最低100,000ゲーム（初期）
- **データ形式**: `.npy`（NumPy Binary）
  ```
  {
    'tokens': object array of int64 arrays (variable length)
    'policies': float32 array [N, action_dim]
    'values': float32 array [N]
    'metadata': dict (optional)
  }
  ```

### 4.2 データ生成スクリプト
```python
# scripts/generate_transformer_data.py
import dm_ai_module as dm

def generate_data(num_games=10000):
    # C++で自己対戦実行
    runner = dm.ParallelRunner(num_threads=8)
    results = runner.run_self_play(
        deck1, deck2, num_games,
        mcts_iterations=1000
    )
    
    # トークン変換
    tokens_list = []
    for state in results.states:
        tokens = dm.TensorConverter.convert_to_sequence(
            state, state.active_player_id, card_db
        )
        tokens_list.append(tokens)
    
    # 保存
    np.savez_compressed(
        'data/transformer_train.npz',
        tokens=np.array(tokens_list, dtype=object),
        policies=results.policies,
        values=results.values
    )
```

---

## 5. 技術的課題と解決策

### 5.1 課題1: Synergy Matrix のスケーラビリティ
**問題**: 1000×1000の密行列は大きい（4MB）

**解決策**:
- 疎行列化（scipy.sparse.csr_matrix）
- ハッシュベースの動的マッピング
- 上位N件のみ保持（N=10000ペア程度）

---

### 5.2 課題2: 可変長シーケンスの効率的処理
**問題**: パディングによるメモリ/計算の無駄

**解決策**:
- Packed sequences（PyTorch）
- Dynamic batching（長さが近いサンプルをグループ化）
- Flash Attention（メモリ効率的なAttention）

---

### 5.3 課題3: 学習の不安定性
**問題**: Transformerは学習が難しい（勾配消失/爆発）

**解決策**:
- Pre-Layer Normalization（norm_first=True）
- Gradient Clipping（max_norm=1.0）
- Warmup学習率スケジュール
- 小さいデータセットでのOverfitting確認

---

## 6. 逆質問リスト（実装前に確認）

### 6.1 アーキテクチャ
- [ ] **Q1**: Synergy Matrixは手動定義で開始？それとも初期学習データから自動生成？
  - **提案**: 初期は手動（既知コンボ10-20ペア）、段階的に自動学習
  
- [ ] **Q2**: CLSトークンの位置は先頭で良い？他の選択肢（末尾、平均プーリング）は？
  - **提案**: 先頭（BERT方式）で開始、実験で比較
  
- [ ] **Q3**: Positional Encodingは学習可能にする？固定sin/cos？
  - **提案**: 学習可能（現行実装通り）、ゲーム特有の位置関係を学習できる

### 6.2 データ
- [ ] **Q4**: 初期学習データはMLP学習時のものを流用可能？
  - **調査必要**: MLPデータにトークンシーケンスが含まれているか確認
  
- [ ] **Q5**: データ拡張は実施する？（カード順シャッフル等）
  - **提案**: Phase 1では未実装、Phase 2で検討

### 6.3 学習
- [ ] **Q6**: バッチサイズは64で良い？メモリに収まる？
  - **検証必要**: 実機でVRAM使用量を測定
  
- [ ] **Q7**: 学習率は1e-4で良い？事前実験は？
  - **提案**: Grid Search（1e-5, 1e-4, 1e-3）でベスト値を決定

### 6.4 評価
- [ ] **Q8**: 評価指標はvs Random勝率のみ？他に必要な指標は？
  - **提案**: vs MLP勝率、ELOレーティング、平均ターン数も追加
  
- [ ] **Q9**: どの時点で本番デプロイ判断？
  - **提案**: vs MLP勝率55%以上 & 24時間安定稼働確認後

---

## 7. リスク管理

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| 学習が収束しない | 高 | 中 | MLPフォールバック、小データセットで事前検証 |
| VRAM不足 | 中 | 低 | バッチサイズ削減、Gradient Checkpointing |
| 推論速度が遅い | 中 | 中 | モデル量子化（INT8）、ONNX変換 |
| Synergy学習が困難 | 低 | 中 | 手動定義で代替 |

---

## 8. マイルストーンと納品物

### Week 2 完了時
- [ ] TransformerDatasetクラス（テスト済み）
- [ ] TransformerTrainerクラス（基本機能）
- [ ] train_transformer.py（実行可能バージョン）
- [ ] ユニットテスト（カバレッジ>80%）
- [ ] 設計ドキュメント更新

### Week 3 完了時
- [ ] C++バインディング拡張（mask対応）
- [ ] パフォーマンステスト結果レポート
- [ ] ベンチマークレポート（vs MLP比較）
- [ ] 学習済みモデル（初期バージョン）
- [ ] デプロイ判断資料

---

## 9. 参考資料

### 9.1 論文
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020 - MuZero)

### 9.2 実装参考
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

### 9.3 内部ドキュメント
- [dm_toolkit/ai/agent/transformer_model.py](../../dm_toolkit/ai/agent/transformer_model.py)
- [src/ai/encoders/tensor_converter.hpp](../../src/ai/encoders/tensor_converter.hpp)
- [docs/02_AI_System_Specs.md](../02_AI_System_Specs.md)

---

**承認者**: _______________  
**承認日**: _______________

**次回レビュー**: 2026年1月16日（Week 2完了時）
