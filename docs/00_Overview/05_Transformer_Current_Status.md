# Transformer実装状況分析と詳細計画

**作成日**: 2026年1月9日  
**ユーザー決定確認**:
- Q1: Synergy Matrix = A（手動定義で開始）✅
- Q2: CLSトークン位置 = A（先頭）✅
- Q3: バッチサイズスケーリング = 8→16→32→64（徐々に大きくする）✅

---

## 1. 現在の実装状況

### 1.1 実装済みコンポーネント

#### ✅ DuelTransformer クラス
**ファイル**: [dm_toolkit/ai/agent/transformer_model.py](../../dm_toolkit/ai/agent/transformer_model.py)  
**進捗**: 95% 完成

```python
# 現在の実装
- __init__: 全パラメータ実装済み
- forward(): ✅ 完全実装
  - Token Embedding
  - Positional Embedding（学習可能）
  - Synergy Bias注入
  - Transformer Encoder（6層、8ヘッド）
  - CLS Token (index=0) からのポーリング
  - Policy Head & Value Head

# 詳細スペック
- d_model: 256
- nhead: 8
- num_layers: 6
- dim_feedforward: 1024
- activation: GELU
- max_len: 512（要修正→200に統一）
```

**課題**:
- `max_len=512` はトークン仕様の `MAX_SEQ_LEN=200` と一致していない
- → **修正必須**: コンストラクタのデフォルト値を 512→200 に変更

#### ✅ SynergyGraph クラス
**ファイル**: [dm_toolkit/ai/agent/synergy.py](../../dm_toolkit/ai/agent/synergy.py)  
**進捗**: 90% 完成

```python
# 現在の実装
class SynergyGraph(nn.Module):
  - vocab_size, embedding_dim=64
  - synergy_embeddings: nn.Embedding(vocab_size, 64)
  - get_bias_for_sequence(): ✅ 完全実装
    - 出力: [Batch, SeqLen, SeqLen] の相性スコア
    - 計算方法: 埋め込みベクトルの内積

# 初期化オプション
- matrix_path から .npy ファイルをロード可能
```

**課題**:
- 現在は **学習可能な埋め込みベクトル** を使用（embedding_dim=64）
- ユーザー決定Q1「手動定義で開始」との整合性確認が必要
  - 手動定義 = 固定値マトリックスをロード
  - 学習可能 = 初期値をランダムから始める

#### ✅ TensorConverter (C++)
**ファイル**: [src/ai/encoders/tensor_converter.hpp](../../src/ai/encoders/tensor_converter.hpp)  
**進捗**: 80% 完成

```cpp
// 実装済みメソッド
- convert_to_sequence(): GameState → トークン列（長さ可変）
- convert_batch_sequence(): 複数 GameState のバッチ処理

// トークン仕様（既実装）
const int MAX_SEQ_LEN = 200;
const int VOCAB_SIZE = 1000;

enum SpecialToken {
  TOKEN_PAD = 0,           // パディング
  TOKEN_SEP = 1,           // セパレータ
  TOKEN_SELF_HAND_START = 2,
  TOKEN_SELF_MANA_START = 3,
  TOKEN_SELF_BATTLE_START = 4,
  TOKEN_SELF_GRAVE_START = 5,
  TOKEN_SELF_SHIELD_START = 6,
  TOKEN_OPP_HAND_START = 7,
  TOKEN_OPP_MANA_START = 8,
  TOKEN_OPP_BATTLE_START = 9,
  TOKEN_OPP_GRAVE_START = 10,
  TOKEN_OPP_SHIELD_START = 11,
  TOKEN_GLOBAL_START = 12,
  TOKEN_CARD_OFFSET = 100  // カードID はこれ以降
};
```

**課題**:
- 現行実装は **[SEP] token first** の形式（BERT形式）
- ユーザー決定Q2「CLS先頭」との統合確認が必要
  - [CLS] [GLOBAL] [SEP] ... の形式への変更検討

#### ✅ DuelDataset & 学習パイプライン
**ファイル**: [dm_toolkit/training/training_pipeline.py](../../dm_toolkit/training/training_pipeline.py)  
**進捗**: 70% 完成

```python
class DuelDataset(Dataset):
  - 引数: tokens, states, policies, values, masks
  - collate_batch(): 可変長トークン列のパディング対応
  - padding_mask 自動生成

# 既存データ形式
- states: 固定長ベクトル（レガシーMLP用）
- tokens: 可変長トークン列（Transformer用）
```

**課題**:
- 学習データに「トークン列」が含まれているか不明確
- 既存の `states` から C++ TensorConverter で動的生成する必要がある可能性

---

## 2. Week 2 実装前のタスク（今日中）

### 2.1 DuelTransformer 微調整（30分）

```python
# transformer_model.py の修正箇所

# 修正1: max_len デフォルト値の統一
- 現在: def __init__(..., max_len: int = 512, ...)
- 修正後: def __init__(..., max_len: int = 200, ...)

# 修正2: forward() のコメント明確化
- CLS トークンはindex 0（検証必須）
```

**実装者**: 今すぐ  
**所要時間**: 30分

---

### 2.2 SynergyGraph の初期化戦略決定（1時間）

ユーザー決定「手動定義で開始」に対応したコード構造:

```python
# 案A: 固定値マトリックス（手動定義）を使用
class SynergyGraph(nn.Module):
    def __init__(self, vocab_size, manual_synergy_path=None):
        # synergy_matrix[i, j] = 固定値
        if manual_synergy_path:
            self.synergy_matrix = nn.Parameter(
                torch.from_numpy(np.load(manual_synergy_path)),
                requires_grad=False  # 固定値
            )
        else:
            # デフォルト: ゼロ初期化（段階的に定義）
            self.synergy_matrix = nn.Parameter(
                torch.zeros(vocab_size, vocab_size),
                requires_grad=False
            )

# 案B: 学習可能な埋め込みベクトル（現行）
# get_bias_for_sequence() で埋め込みの内積を計算
```

**判定**: 案A + 案B の混合
- 初期化: 手動定義（案A）
- 訓練中: 埋め込みを段階的に学習（案B）
- 実装方法: `SynergyGraph.__init__()` に `trainable: bool` フラグを追加

**実装タスク**:
1. 手動定義ファイルのフォーマット決定（JSON? NPY?）
2. SynergyGraph に `freeze/unfreeze` メソッド追加
3. サンプルマトリックス（10-20ペア）を作成

---

### 2.3 既存学習データの確認（1時間）

**タスク**: データフォーマット確認スクリプト実行

```python
# python/inspect_training_data.py を作成して実行
import numpy as np
import os

data_path = "data/training_data.npz"
if os.path.exists(data_path):
    data = np.load(data_path)
    print("Keys:", list(data.files))
    for key in data.files:
        print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
else:
    print("No training_data.npz found")
    # 検索
    import glob
    matches = glob.glob("data/**/training*.npz", recursive=True)
    print("Found:", matches)
```

**結果判定**:
- ✅ `tokens` キーが存在 → データ流用可能（作業 2時間）
- ✅ `states` のみ → TensorConverter で動的変換（作業 3時間）
- ❌ データが古い/破損 → 新規生成（作業 8時間）

---

## 3. Week 2 Day 1 具体的タスク（1月13日）

### 3.1 Synergy マトリックス（手動定義）の作成

**ファイル作成**: `data/synergy_matrix_v1.json`

```json
{
  "description": "Manual Synergy Matrix (v1) - Card Combo Pairs",
  "version": "1.0",
  "pairs": [
    {
      "name": "Revolution Change Combo",
      "cards": ["《勝利宣言 鬼丸「覇」》", "《多色カード X》"],
      "synergy_score": 0.8,
      "description": "多色カードを踏み台にして革命チェンジ"
    },
    {
      "name": "Spell Chain",
      "cards": ["《呪文A》", "《呪文B》"],
      "synergy_score": 0.6
    }
    // ... 10-20ペア
  ]
}
```

**実装**:
1. カード相性ペアを JSON で定義
2. `SynergyGraph.load_from_json()` メソッド実装
3. NumPy 行列 → PyTorch テンソルに変換

**チェックリスト**:
- [ ] 10-20ペアの定義完了
- [ ] JSON パース実装
- [ ] 行列サイズ 1000×1000 で初期化
- [ ] テスト: `test_synergy_loading.py`

---

### 3.2 データパイプラインの統合

**目標**: GameState → Token列 → Transformer の自動変換

```python
# train_transformer.py (新規作成)

# Step 1: GameState をトークン化（C++ TensorConverter 経由）
game_states = load_scenario_data()  # 1000ゲーム
token_sequences = convert_to_sequence_batch(game_states)  # → [1000, 200]

# Step 2: DuelDataset に保存
dataset = DuelDataset(
    tokens=token_sequences,
    states=None,  # Transformerなので不要
    policies=policies,
    values=values
)

# Step 3: DataLoader でバッチ処理
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch)

# Step 4: 学習ループ
for epoch in range(epochs):
    for batch in loader:
        tokens = batch['tokens']  # [8, 200]
        padding_mask = batch['padding_mask']  # [8, 200]
        policy_targets = batch['policy']  # [8, action_dim]
        value_targets = batch['value']  # [8, 1]
        
        # Forward pass
        policy_logits, value_pred = model(tokens, padding_mask)
        
        # Loss computation
        policy_loss = F.cross_entropy(policy_logits, policy_targets.argmax(dim=1))
        value_loss = F.mse_loss(value_pred, value_targets)
        
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()
```

---

## 4. 指標化・実装判定基準

### 4.1 Week 2 Day 1 の成功基準

| 項目 | 基準 | 確認方法 |
|------|------|---------|
| Synergy 手動定義 | 10-20ペア実装 | `len(synergy_matrix.nonzero()) >= 10` |
| Token 生成 | 1000サンプルで可変長200以下 | `max(len(seq) for seq in tokens) <= 200` |
| Dataset 作成 | padding_mask が正確に生成 | `test_dataset_masks.py` ✅ |
| DataLoader | バッチサイズ 8 で正常に動作 | バッチシェイプ `[8, 200]` |

### 4.2 Week 2 Day 2-3 マイルストーン

| 日付 | マイルストーン | 成功基準 |
|------|---------------|---------|
| 1月14日 | モデル初期化 | `model = DuelTransformer(...)` → forward() ✅ |
| 1月15日 | 訓練ループ開始 | 100エポックで Loss 低下確認 |
| 1月16日 | 基本的な過学習テスト | Training loss vs Val loss グラフ作成 |

---

## 5. 修正が必要な箇所（優先順）

### 🔴 Critical（今日中）

| No. | 項目 | ファイル | 修正内容 | 優先度 |
|-----|------|---------|---------|--------|
| 1 | max_len 統一 | transformer_model.py | 512→200 | 🔴 Critical |
| 2 | Synergy初期化戦略 | synergy.py | 手動定義用コード追加 | 🔴 Critical |
| 3 | データ形式確認 | inspect_training_data.py | スクリプト実行 | 🔴 Critical |

### 🟡 High（Week 2 Day 1）

| No. | 項目 | ファイル | 修正内容 | 優先度 |
|-----|------|---------|---------|--------|
| 4 | Synergy JSON ロード | synergy.py | `load_from_json()` メソッド | 🟡 High |
| 5 | CLS token 検証 | tensor_converter.cpp | シーケンス先頭に [CLS] 挿入 | 🟡 High |
| 6 | train_transformer.py | training_pipeline.py | Transformer専用学習スクリプト | 🟡 High |

---

## 6. 実装予定時間表

```
今日（1月9日）
├─ 10:00 DuelTransformer max_len 修正 (30分)
├─ 10:30 SynergyGraph 初期化戦略決定 (30分)
├─ 11:00 データ形式確認スクリプト実行 (30分)
└─ 11:30 実装計画レビュー＆質問対応

Week 2 Day 1（1月13日）
├─ Synergy マトリックス手動定義 (2時間)
├─ Token 生成パイプライン (2時間)
├─ Dataset & DataLoader 統合 (1時間)
└─ テスト実装 (1時間)

Week 2 Day 2-3（1月14-16日）
├─ DuelTransformer 初期化・forward() テスト
├─ 訓練ループ実装
├─ Loss 曲線の確認
└─ ハイパーパラメータ微調整（バッチサイズ段階的拡大）
```

---

## 7. 残りの逆質問（Q4-Q9）の最終化

ユーザーが既に回答した内容を反映：

| 質問 | ユーザー決定 | 実装への影響 |
|------|----------|-----------|
| Q1: Synergy初期化 | **A（手動定義）** | 固定値行列をサポート |
| Q2: CLSトークン位置 | **A（先頭）** | [CLS] [GLOBAL] [SEP] ... の形式 |
| Q3: バッチサイズ | **8→16→32→64** | DataLoader の batch_size スケジューリング |
| Q4: データ流用 | **未確認** | 本日スクリプト実行で判定 |
| Q5: Pos Encoding | **未決定** | 学習可能を推奨（現行実装） |
| Q6: データ拡張 | **未決定** | Phase 2 で実装推奨 |
| Q7: 評価指標 | **未決定** | vs Random, vs MLP, 推奨 |
| Q8: デプロイ基準 | **未決定** | vs MLP ≥ 55% 推奨 |
| Q9: Synergy行列 | **未決定** | 密行列OK（4MB小さい） |

**次のステップ**: Q4-Q9 の最終回答を待つ前に、上記3つの Critical タスク を完了可能

---

## 📋 実装開始のチェックリスト

- [ ] **本日完了項目**
  - [ ] DuelTransformer の max_len 修正
  - [ ] SynergyGraph の初期化戦略コード化
  - [ ] データ形式確認スクリプト実行

- [ ] **Week 2 開始前確認**
  - [ ] Q4: データ流用可能性の最終判定
  - [ ] Q5-Q6: Pos Encoding & データ拡張の決定
  - [ ] GPU メモリ測定（バッチサイズ 8 の場合）

- [ ] **Week 2 実装開始条件**
  - [ ] Synergy マトリックス定義完了
  - [ ] トークン生成パイプライン検証
  - [ ] 学習スクリプト（train_transformer.py）準備完了
