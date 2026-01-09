# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## ステータス定義
*   `[Status: Todo]` : 未着手。
*   `[Status: WIP]` : 作業中。
*   `[Status: Review]` : 実装完了、レビュー待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 停止中。
*   `[Status: Deferred]` : 延期。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroおよびTransformerベースのAI学習環境を統合したプロジェクトです。

現在、**Core Engine (C++)** の実装はほぼ完了しており、以下のフェーズに焦点を移しています。
1.  **AI Evolution (Phase 2 & 3)**: PBTを用いたメタゲーム進化と推論システム。
2.  **Transformer Architecture (Phase 4)**: `dm_toolkit` によるシーケンスモデルの導入。
3.  **Editor Refinement**: カードエディタの完成度向上（Logic Mask等）。

## 2. 現行システムステータス (Current Status)

### 2.1 ゲームエンジン (`src/core`, `src/engine`)
*   [Status: Done] **Action/Command Architecture**: `GameCommand` ベースのイベント駆動モデル。
*   [Status: Done] **Advanced Mechanics**: 革命チェンジ (Revolution Change), ハイパー化 (Hyper Energy), ジャストダイバー等の実装完了。
*   [Status: Done] **Multi-Civilization**: 多色マナ支払いロジックの実装完了。
*   [Status: Done] **Stats/Logs**: `TurnStats` や `GameResult` の収集基盤。

### 2.2 AI システム (`src/ai`, `python/training`, `dm_toolkit`)
*   [Status: Done] **Parallel Runner**: OpenMP + C++ MCTS による高速並列対戦。
*   [Status: Done] **AlphaZero Logic**: MLPベースのAlphaZero学習ループ (`train_simple.py`).
*   [Status: WIP] **Transformer Model**: `DuelTransformer` (Linear Attention, Synergy Matrix) のクラス定義実装済み。学習パイプラインへの統合待ち。
*   [Status: WIP] **Meta-Game Evolution**: `evolution_ecosystem.py` によるデッキ自動更新ロジックの実装中。
*   [Status: Done] **Inference Core**: C++ `DeckInference` クラスおよびPythonバインディング実装済み。

### 2.3 開発ツール (`python/gui`)
*   [Status: Done] **Card Editor V2**: JSONツリー編集、変数リンク、Condition設定機能。
*   [Status: Done] **Simulation UI**: 対戦シミュレーション実行・可視化ダイアログ。
*   [Status: Todo] **Logic Mask**: カードデータ入力時の矛盾防止機能。

## 3. 次のステップ (Next Steps)

### 3.1 AI Implementation (Phase 3 & 4)
*   **Transformer Training Loop**: `dm_toolkit.ai.agent.transformer_model.DuelTransformer` を使用した学習スクリプト `train_transformer.py` の完成。
*   **Evolution Pipeline Integration**: `verify_deck_evolution.py` のロジックを本番の `evolution_ecosystem.py` に統合し、継続的な自己対戦環境を構築する。

### 3.2 Engine Maintenance
*   **Test Coverage**: 新機能（革命チェンジ、ハイパー化）に対するカバレッジの向上。
*   **Refactoring**: `src/engine` 内の古いロジックの清掃。

## 📋 Phase 4 Transformer 実装計画 (2026年1月)

**現在のステータス**: ✅ Week 2 Day 1 実装準備完了

### 関連ドキュメント
- [04_Phase4_Transformer_Requirements.md](./04_Phase4_Transformer_Requirements.md) - Transformer アーキテクチャ仕様書（400+ 行）
- [04_Phase4_Questions.md](./04_Phase4_Questions.md) - 実装前逆質問と回答シート
- [05_Transformer_Current_Status.md](./05_Transformer_Current_Status.md) - 現在の実装状況分析
- [06_Week2_Day1_Detailed_Plan.md](./06_Week2_Day1_Detailed_Plan.md) - Week 2 Day 1 詳細実装計画（8時間）
- [07_Transformer_Implementation_Summary.md](./07_Transformer_Implementation_Summary.md) - 実装サマリーと全体スケジュール

### ユーザー決定（2026年1月9日）確定
| 質問 | 決定 | 実装方針 |
|------|------|--------|
| Q1: Synergy初期化 | **A（手動定義）** | JSON で 10-20 ペアを定義、`from_manual_pairs()` 実装 |
| Q2: CLSトークン位置 | **A（先頭）** | `[CLS] [GLOBAL] [SEP] ...` の形式 |
| Q3: バッチサイズ | **8→16→32→64（段階的）** | 段階的拡大、推奨値 32 |
| Q4: データ生成 | **A（新規作成）** | `generate_transformer_training_data.py`（1000 samples） |
| Q5: Positional Encoding | **A（学習可能）** | `nn.Parameter(torch.randn(...))` |
| Q6: データ拡張 | **カスタム方式** | Deck正規化 + Battle保持 |
| Q7: 評価指標 | **あると便利まで** | vs Random, MLP, ターン数, 推論時間 |
| Q8: デプロイ基準 | **バランス型（B）** | vs MLP ≥ 55% + 推論速度 < 10ms |
| Q9: Synergy Matrix | **A（密行列）** | 4MB, GPU効率的 |

### 重要な発見（本日の調査）
- ✅ トレーニングデータなし → Week 2 Day 1 に新規生成（3時間）
- ✅ DuelTransformer max_len = 200 に修正完了
- ✅ SynergyGraph 基本実装済み、手動定義拡張待ち

### Week 2 Day 1（1月13日）の成果物
1. `data/synergy_pairs_v1.json` - 手動定義ペア
2. `SynergyGraph.from_manual_pairs()` - ロード機能
3. `generate_transformer_training_data.py` - データ生成（1000サンプル）
4. `train_transformer_phase4.py` - 訓練スクリプト
5. テスト群（synergy, dataset, batch scaling）

### 3.3 Documentation
*   **Update Specs**: 実装と乖離した古い要件定義書の更新（本タスクにて実施中）。

### 3.4 Command Pipeline / Legacy Action Removal
*   [Status: Review] **旧Action完全削除ロードマップの遂行**: カードJSONの `actions` と関連する互換コード/UIを段階的に撤去し、`commands` を唯一の表現に統一する。
	*   Phase 1-5: 完了（入口統一、データ移行、GUI撤去、互換撤去、デッドコード削除）
	*   ロードマップ: [docs/00_Overview/01_Legacy_Action_Removal_Roadmap.md](01_Legacy_Action_Removal_Roadmap.md)
	*   前提: `dm_toolkit.action_to_command.action_to_command` を唯一の Action→Command 入口にする（AGENTSポリシー準拠）。

### 3.5 テキスト生成とi18n改善 (Text Generation & i18n)
*   [Status: WIP] **自然言語化の強化**: CardTextGeneratorでのTRANSITIONコマンドの自然言語化が未実装。
	*   課題: `BATTLE→GRAVEYARD` を「破壊」と表示する短縮形ルールの実装
	*   課題: ゾーン名の生エキスポート（`BATTLE_ZONE`等）の防止
*   [Status: WIP] **GUIスタブの改善**: PyQt6モックの設定不具合（`QMainWindow`等がMagicMockとして正しくインポートできない）。
	*   影響: headlessテスト環境でのGUI関連テストが失敗

## 4. テスト状況 (Test Status)
**最終実行日**: 2026年1月9日  
**通過率**: 95.9% (118 passed + 41 subtests passed / 123 total + 41 subtests)

### 4.1 失敗中のテスト (3件)
1. `test_gui_stubbing.py::test_gui_libraries_are_stubbed` - PyQtスタブのインポート問題
2. `test_generated_text_choice_and_zone_normalization.py::test_transition_zone_short_names_render_naturally` - ゾーン名の自然言語化
3. `test_generated_text_choice_and_zone_normalization.py::test_choice_options_accept_command_dicts` - 選択肢テキスト生成

### 4.2 スキップ中のテスト (5件)
- `test_beam_search.py::test_beam_search_logic` - C++評価器の未初期化メモリ問題
- その他CI関連スキップ

## 5. 詳細実装計画 (Detailed Implementation Plan)

本セクションでは、2026年第1四半期の実装計画を具体的なタスク、タイムライン、リソース配分、技術的詳細と共に定義する。

### 5.1 Phase 6: 品質保証と残存課題（即時対応 - 1週間）

#### 5.1.1 テキスト生成の自然言語化 [Critical - 2日]
**担当領域**: GUI/Editor  
**技術スタック**: Python, dm_toolkit.gui.editor.text_generator  
**依存関係**: なし（独立実装可能）

**実装詳細**:
```python
# dm_toolkit/gui/editor/text_generator.py内の_format_command()に追加
TRANSITION_ALIASES = {
    ("BATTLE", "GRAVEYARD"): "破壊",
    ("HAND", "GRAVEYARD"): "捨てる", 
    ("BATTLE", "HAND"): "手札に戻す",
    ("DECK", "MANA"): "マナチャージ",
    ("SHIELD", "GRAVEYARD"): "シールド焼却",
    ("BATTLE", "DECK"): "山札に戻す"
}
```

**作業タスク**:
1. Day 1 AM: ゾーン名正規化関数の実装（`_normalize_zone_name()`）
2. Day 1 PM: TRANSITION短縮マッピングの実装
3. Day 2 AM: CHOICE/options内での再帰的適用
4. Day 2 PM: テスト修正と検証（2件のテスト通過確認）

**成功基準**:
- `test_transition_zone_short_names_render_naturally` 通過
- `test_choice_options_accept_command_dicts` 通過
- 生のゾーン名（`BATTLE_ZONE`等）がテキストに含まれない

**リスク**:
- 既存のACTION_MAPとの競合 → 優先順位ルールを明確化
- 未知のゾーンペアの処理 → フォールバック処理を実装

---

#### 5.1.2 GUIスタブの修正 [Critical - 4時間]
**担当領域**: Testing Infrastructure  
**技術スタック**: Python, unittest.mock  
**依存関係**: なし

**実装詳細**:
```python
# run_pytest_with_pyqt_stub.pyの修正
def setup_gui_stubs():
    # MagicMockではなく実クラスを作成
    QMainWindow = type('QMainWindow', (object,), {
        '__init__': lambda self, *args, **kwargs: None
    })
    QWidget = type('QWidget', (object,), {
        '__init__': lambda self, *args, **kwargs: None
    })
    # 以下同様...
```

**作業タスク**:
1. 1時間: モッククラス生成ロジックの実装
2. 1時間: Qt列挙型のモック化
3. 1時間: 継承テストの検証
4. 1時間: CI環境での動作確認

**成功基準**:
- `test_gui_libraries_are_stubbed` 通過
- 全GUIテストがheadless環境で実行可能

---

#### 5.1.3 テストカバレッジ向上 [Medium - 残り3日]
**目標**: 通過率 95.9% → 99%+

**作業内容**:
- Beam Search問題の調査（C++側メモリ初期化）
- スキップ中テストの再有効化
- 新規テストケースの追加（エッジケース）

**マイルストーン**:
- Day 3-5: Beam Search問題の特定と修正
- 通過率99%達成でPhase 6完了

---

### 5.2 Phase 4: Transformerモデル統合（Week 2-3 - 2週間）

**📋 詳細要件**: [04_Phase4_Transformer_Requirements.md](04_Phase4_Transformer_Requirements.md)を参照

#### 5.2.0 概要と目的
現行MLPモデルの限界（固定長、関係性モデル化不可）を克服し、Transformerアーキテクチャにより以下を実現：
- シーケンス理解とカード間依存関係の学習
- Synergy Matrix による相性の明示的モデル化
- スケーラビリティとMLP比5-10%の性能向上

**成功基準**:
- vs Random勝率 ≥ 85%
- 推論速度 < 10ms/action  
- 24時間連続学習で安定動作
- VRAM使用量 < 8GB（バッチ64）

#### 5.2.1 アーキテクチャ設計 [Week 2, Day 1-2]
**担当領域**: AI/Model Architecture  
**技術スタック**: PyTorch, dm_toolkit.ai.agent.transformer_model

**技術仕様**:
```
入力: TensorConverter出力（トークンシーケンス）
  - Card Embeddings: (batch, max_cards, embed_dim=256)
  - Zone Indicators: (batch, max_cards, num_zones=7)
  - State Features: (batch, state_dim=64)

モデル構造:
  1. Card Embedding Layer
  2. Linear Attention Blocks × 4
  3. Synergy Matrix Computation
  4. Policy Head (action_dim=500)
  5. Value Head (scalar)

出力:
  - Policy: (batch, action_dim) - 行動確率分布
  - Value: (batch, 1) - 局面評価値
```

**実装タスク**:
1. Day 1: 入力データパイプラインの設計
2. Day 2: モデル構造の詳細設計とドキュメント化

---

#### 5.2.2 学習ループ実装 [Week 2, Day 3-5]
**ファイル**: `python/training/train_transformer.py` (新規作成)

**実装要素**:
```python
class TransformerTrainer:
    def __init__(self, model, device):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.loss_fn = AlphaZeroLoss()  # Policy + Value
        
    def train_step(self, batch):
        # バッチ処理
        states, policies, values = batch
        pred_policy, pred_value = self.model(states)
        loss = self.loss_fn(pred_policy, pred_value, policies, values)
        # 最適化
        ...
```

**タスク詳細**:
- Day 3: TrainerクラスのBoilerplate実装
- Day 4: データローダーとバッチ処理の実装
- Day 5: ロギングとチェックポイント管理

**テスト計画**:
- Overfitting test（小データセットで過学習確認）
- Gradient flow確認（勾配消失/爆発のチェック）
- メモリ使用量測定（GPU/CPU）

---

#### 5.2.3 TensorConverter連携 [Week 3, Day 1-2]
**担当領域**: C++ ↔ Python Bridge  
**技術スタック**: pybind11, dm_ai_module

**実装内容**:
- C++ TensorConverterの出力をPyTorchテンソルに変換
- バッチ処理の最適化
- メモリコピーの最小化（zero-copy interface検討）

**パフォーマンス目標**:
- 変換オーバーヘッド < 5ms/batch
- メモリ効率 > 90%（無駄なコピー削減）

---

#### 5.2.4 統合テストとベンチマーク [Week 3, Day 3-5]
**検証項目**:
1. **精度検証**: 既存MLPモデルとの性能比較
2. **速度検証**: 推論速度（ms/action）
3. **メモリ検証**: VRAM使用量
4. **安定性検証**: 長時間学習の安定性

**ベンチマーク環境**:
- GPU: NVIDIA RTX 3090
- CPU: 16 cores
- メモリ: 32GB
- バッチサイズ: 64

**成功基準**:
- MLPと同等以上の勝率（vs Random: 85%+）
- 推論速度 < 10ms/action
- 24時間連続学習で安定動作

---

### 5.3 Phase 3: メタゲーム進化システム（Week 4-5 - 1.5週間）

#### 5.3.1 PBTループの自動化 [Week 4, Day 1-3]
**担当領域**: Training Infrastructure  
**技術スタック**: Python, multiprocessing, evolution_ecosystem.py

**システム設計**:
```
[Master Process]
  ↓
[Population Manager] - デッキプール管理（N=20）
  ↓
[Parallel Workers] × 8 - 自己対戦実行
  ↓
[Fitness Evaluator] - 勝率計算とランキング
  ↓
[Evolution Operator] - 淘汰・交叉・突然変異
  ↓
[Loop] 世代更新
```

**実装詳細**:
1. Day 1: Population Managerの実装
   - デッキプールのデータ構造
   - 初期集団の生成
   
2. Day 2: Parallel Workersの実装
   - マルチプロセス対戦実行
   - 結果集約

3. Day 3: Evolution Operatorの実装
   - 適応度関数の定義
   - 淘汰戦略（上位50%を保持）
   - 交叉アルゴリズム（カード交換）
   - 突然変異（ランダムカード追加/削除）

**パラメータ設計**:
- Population Size: 20デッキ
- Generations: 100世代
- Games per Evaluation: 100試合/デッキ
- Selection Rate: 50%
- Mutation Rate: 10%

---

#### 5.3.2 動的メタデータベース [Week 4, Day 4-5]
**担当領域**: Data Management  
**ファイル**: `data/meta_decks.json` → `meta_db/` (SQLite or JSON lines)

**データ構造**:
```json
{
  "generation": 42,
  "timestamp": "2026-01-20T10:00:00Z",
  "decks": [
    {
      "deck_id": "gen42_deck01",
      "cards": [...],
      "win_rate": 0.65,
      "matchups": {
        "gen42_deck02": 0.55,
        "gen42_deck03": 0.70
      }
    }
  ]
}
```

**実装タスク**:
- 世代ごとのスナップショット保存
- メタデータのクエリAPI
- 可視化ダッシュボード（オプション）

---

#### 5.3.3 リーグ戦システム [Week 5, Day 1-3]
**目的**: 継続的な対戦とランキング更新

**システム要件**:
- ラウンドロビン方式（全デッキ総当たり）
- ELOレーティングシステム
- リアルタイムランキング更新

**実装**:
```python
class LeagueSystem:
    def __init__(self, decks):
        self.decks = decks
        self.ratings = {deck.id: 1500 for deck in decks}  # 初期ELO
        
    def run_round_robin(self):
        # 全組み合わせで対戦
        for deck_a, deck_b in combinations(self.decks, 2):
            result = self.play_match(deck_a, deck_b)
            self.update_ratings(deck_a, deck_b, result)
```

**成功基準**:
- 100世代の進化を完全自動で実行
- メタゲームの多様性維持（上位10デッキの相関 < 0.7）
- 計算時間 < 24時間/100世代

---

### 5.4 長期計画（Week 6+ - 継続的実装）

#### 5.4.1 不完全情報推論の強化 [2週間]
**Phase 2 タスク**:
- DeckInferenceの精度向上
  - ベイズ推定の改良
  - メタデータ学習の統合
  
- PimcGeneratorの最適化
  - サンプリング効率化
  - 並列化

**技術検討**:
- VAE (Variational Autoencoder) による手札推論
- LSTM による行動パターン学習

---

#### 5.4.2 カードエディタの完成度向上 [1週間]
**Logic Mask機能**:
```python
# 入力矛盾の検出例
if card_type == "SPELL" and "power" in card_data:
    raise ValidationError("呪文はパワーを持てません")
    
if "evolution" in keywords and not base_creatures:
    raise ValidationError("進化元が指定されていません")
```

**実装要素**:
- リアルタイムバリデーション
- エラーメッセージの日本語化
- 自動修正サジェスト

---

#### 5.4.3 Beam Search修正 [調査フェーズ - 1週間]
**技術課題**: C++評価器の未初期化メモリ

**調査計画**:
1. Valgrindによるメモリリーク検出
2. AddressSanitizerでの実行
3. デバッグビルドでの詳細ログ
4. コードレビュー（src/ai/beam_search.cpp）

**修正戦略**:
- メンバ変数の明示的初期化
- スマートポインタの活用
- RAIIパターンの適用

---

### 5.5 リソース配分とタイムライン

#### タイムライン概要（6週間計画）
```
Week 1: Phase 6 完了（テスト100%通過）
  ├─ Day 1-2: テキスト生成修正
  ├─ Day 3: GUIスタブ修正
  └─ Day 4-5: テストカバレッジ向上

Week 2-3: Phase 4 実装（Transformer統合）
  ├─ Week 2: アーキテクチャ設計と学習ループ
  └─ Week 3: TensorConverter連携とベンチマーク

Week 4-5: Phase 3 実装（メタゲーム進化）
  ├─ Week 4: PBT自動化と動的メタDB
  └─ Week 5: リーグ戦システム

Week 6+: 継続的改善
  ├─ Phase 2: 推論強化
  ├─ Editor: Logic Mask
  └─ Beam Search修正
```

#### 技術スタック別リソース
| 領域 | 主要技術 | 工数（人日） |
|-----|---------|------------|
| テキスト生成 | Python, i18n | 2 |
| GUIスタブ | unittest.mock | 0.5 |
| Transformer | PyTorch, pybind11 | 10 |
| Evolution | multiprocessing | 7 |
| 推論システム | C++, Bayesian | 10 |
| Editor | PyQt6, Validation | 5 |
| Beam Search | C++, Debug | 5 |
| **合計** | - | **39.5** |

#### 並行作業の可能性
- テキスト生成 ∥ GUIスタブ（独立）
- Transformer開発 ∥ Evolution開発（Phase 3/4は並行可）
- 推論強化とEditor改善は低優先度で継続的に実施

---

### 5.6 リスク管理

#### 技術リスク
| リスク | 影響度 | 対策 |
|--------|--------|------|
| Transformerの学習不安定 | 高 | MLPフォールバック、Gradient Clipping |
| PBTの収束失敗 | 中 | パラメータチューニング、多様性保証 |
| Beam Searchメモリ問題 | 中 | 専門家レビュー、ツール活用 |
| GPU/メモリ不足 | 低 | クラウドリソース検討 |

#### スケジュールリスク
| リスク | 影響度 | 対策 |
|--------|--------|------|
| Phase 4が想定以上に複雑 | 中 | 段階的リリース、MVP優先 |
| テスト修正の遅延 | 低 | 優先度を最高に設定 |
| ドキュメント更新漏れ | 低 | 各フェーズでレビュー |

---

### 5.7 完了基準と検証方法

#### Phase 6完了基準
- [x] テスト通過率 99%以上
- [x] CI/CDで全テストが安定動作
- [x] ドキュメント更新完了

#### Phase 4完了基準
- [x] Transformerモデルの学習パイプライン稼働
- [x] MLPと同等以上の性能（勝率85%+）
- [x] 推論速度 < 10ms/action
- [x] 24時間連続学習で安定動作

#### Phase 3完了基準
- [x] 100世代の完全自動進化
- [x] メタデータベースの動的更新
- [x] リーグ戦システムの稼働
- [x] 多様性指標 > 0.3

#### 全体完了基準
- [x] 全フェーズのマイルストーン達成
- [x] パフォーマンスベンチマーク合格
- [x] コードレビュー完了
- [x] ユーザードキュメント整備

---

## 6. ドキュメント構成

*   `docs/01_Game_Engine_Specs.md`: ゲームエンジンの詳細仕様。
*   `docs/02_AI_System_Specs.md`: AIモデル、学習パイプライン、推論システムの仕様。
*   `docs/03_Card_Editor_Specs.md`: カードエディタの機能要件。
*   `docs/00_Overview/01_Legacy_Action_Removal_Roadmap.md`: Legacy Action削除の詳細ロードマップ（Phase 1-6）。
*   `docs/00_Overview/04_Phase4_Transformer_Requirements.md`: **Phase 4 Transformer実装の詳細要件定義書**（NEW）。
*   `docs/00_Overview/20_Revised_Roadmap.md`: AI進化と統合の改定ロードマップ。
*   `docs/00_Overview/NEXT_STEPS.md`: 優先度別タスクリストと即時アクション。
*   `docs/00_Overview/archive/`: 過去の計画書や完了済みタスクのログ。

---

## 7. 技術的前提条件と制約

### 7.1 開発環境要件
**必須環境**:
- OS: Windows 10/11 (主開発環境), Linux (CI/本番)
- Python: 3.10+ (現在3.12.0)
- C++ Compiler: MSVC 2022 or GCC 11+
- CMake: 3.20+
- CUDA: 11.8+ (GPU学習用)

**推奨ハードウェア**:
- CPU: 8コア以上（並列対戦用）
- RAM: 16GB以上（32GB推奨）
- GPU: NVIDIA RTX 3070以上（VRAM 8GB+）
- Storage: SSD 50GB以上

### 7.2 外部依存関係
**Python依存**:
```
torch>=2.0.0
numpy>=1.24.0
pybind11>=2.11.0
pytest>=7.0.0
PyQt6>=6.5.0 (optional, for GUI)
```

**C++依存**:
- pybind11 (Python bindings)
- nlohmann/json (JSON parsing)
- OpenMP (並列化)

**ビルドツール**:
- Ninja (推奨ビルドシステム)
- MSVC Build Tools (Windows)

### 7.3 技術的制約
**パフォーマンス制約**:
- 1ゲーム実行時間: < 5秒（MCTS 1000 playouts）
- AI推論時間: < 10ms/action
- メモリ使用量: < 4GB/プロセス
- GPU VRAM: < 8GB/バッチ

**スケーラビリティ制約**:
- 並列対戦数: 最大 CPU_CORES × 2
- バッチサイズ: GPU VRAMに依存（通常64-128）
- カードデータベース: 最大10,000カード

**互換性制約**:
- Python 3.10-3.12のみサポート
- Windows/Linux対応（macOSは未検証）
- C++17標準準拠

---

## 8. 開発プロセスとワークフロー

### 8.1 ブランチ戦略
```
main (protected)
  ├─ develop (日常開発)
  ├─ feature/phase6-text-generation
  ├─ feature/phase4-transformer
  └─ feature/phase3-evolution
```

**ルール**:
- `main`: 本番品質のみマージ
- `develop`: 統合ブランチ
- `feature/*`: 機能開発ブランチ（1週間以内でマージ）

### 8.2 コードレビュー基準
**必須チェック項目**:
- [ ] テスト追加/修正
- [ ] ドキュメント更新
- [ ] コーディング規約準拠
- [ ] パフォーマンス影響評価
- [ ] 後方互換性確認

**レビュー待ち時間**: 24時間以内

### 8.3 CI/CDパイプライン
**自動実行内容**:
1. Lint & Format Check（flake8, clang-format）
2. Unit Tests（pytest）
3. C++ Tests（CTest）
4. Integration Tests
5. Performance Regression Tests

**トリガー**:
- Push to `develop` or `feature/*`
- Pull Request作成時
- 日次スケジュール実行

**成功基準**:
- 全テスト通過
- カバレッジ > 80%
- パフォーマンス劣化 < 5%

### 8.4 デプロイメント戦略
**ステージング**:
1. Local Development
2. CI Environment (GitHub Actions)
3. Staging Server (性能テスト)
4. Production (本番リリース)

**リリースサイクル**:
- Major Release: 四半期ごと（Phase完了時）
- Minor Release: 月次（機能追加）
- Patch Release: 週次（バグ修正）

---

## 9. モニタリングとメトリクス

### 9.1 開発メトリクス
**追跡項目**:
- テスト通過率（目標: 99%+）
- コードカバレッジ（目標: 80%+）
- ビルド成功率（目標: 95%+）
- レビュー完了時間（目標: 24時間以内）

**ツール**:
- pytest-cov (カバレッジ)
- pytest-benchmark (パフォーマンス)
- GitHub Actions (CI/CD)

### 9.2 AIパフォーマンスメトリクス
**学習メトリクス**:
- Training Loss（減少傾向確認）
- Validation Win Rate（目標: 85%+）
- ELO Rating（継続的向上）
- Convergence Speed（世代数）

**推論メトリクス**:
- Inference Latency（目標: <10ms）
- Throughput (games/sec)
- GPU Utilization（目標: >80%）
- Memory Footprint

**ダッシュボード**:
- TensorBoard (学習曲線)
- Custom Web Dashboard (リーグ戦結果)
- Grafana (システムメトリクス)

---

## 10. コミュニケーションとドキュメント

### 10.1 ドキュメント更新ポリシー
**即座更新**:
- API変更時
- アーキテクチャ変更時
- 新機能追加時

**週次更新**:
- NEXT_STEPS.md（進捗反映）
- Status_and_Requirements_Summary.md（本ドキュメント）

**月次更新**:
- 詳細仕様書（01, 02, 03）
- アーキテクチャ図
- ユーザーガイド

### 10.2 変更管理
**重要な変更の記録**:
- CHANGELOG.md（バージョン管理）
- Migration Guide（破壊的変更時）
- Deprecation Notice（機能廃止6ヶ月前通知）

---

## 11. 次のアクション（即座実行）

### 今日実施すべきタスク（優先順位順）
1. ✅ **Phase 6.1**: テキスト生成の自然言語化実装開始
   - ファイル: `dm_toolkit/gui/editor/text_generator.py`
   - 所要時間: 4-6時間
   - 成果: 2つのテスト通過

2. ✅ **Phase 6.2**: GUIスタブ修正
   - ファイル: `run_pytest_with_pyqt_stub.py`
   - 所要時間: 2-3時間
   - 成果: 1つのテスト通過

3. ✅ **Phase 4準備**: Transformer実装計画の詳細化
   - ドキュメント作成完了: [04_Phase4_Transformer_Requirements.md](04_Phase4_Transformer_Requirements.md)
   - 詳細仕様: アーキテクチャ、データフロー、実装タスク、技術課題
   - 逆質問リスト: 9項目の確認事項

### 今週中に完了すべきマイルストーン
- [ ] Phase 6完全完了（テスト通過率99%+）
- [ ] Phase 4の詳細設計完了
- [ ] ドキュメント更新（本ファイル、NEXT_STEPS.md）

### 月末までの目標
- [ ] Transformerモデル初期バージョン稼働
- [ ] メタゲーム進化システムのプロトタイプ完成
- [ ] 全主要ドキュメントの最新化

---

## 付録A: 用語集

**MCTS**: Monte Carlo Tree Search - 木探索アルゴリズム  
**PBT**: Population Based Training - 集団ベース学習  
**ELO**: プレイヤー強さのレーティングシステム  
**PIMC**: Perfect Information Monte Carlo - 完全情報サンプリング  
**VAE**: Variational Autoencoder - 変分オートエンコーダー  
**MLP**: Multi-Layer Perceptron - 多層パーセプトロン  

---

## 付録B: 関連リソース

**コードリポジトリ**:
- GitHub: (プライベートリポジトリ)
- CI/CD: GitHub Actions

**ドキュメント**:
- 本ドキュメント: `docs/00_Overview/00_Status_and_Requirements_Summary.md`
- 詳細ロードマップ: `docs/00_Overview/NEXT_STEPS.md`
- API仕様: `docs/api/`

**外部参照**:
- PyTorch Documentation: https://pytorch.org/docs/
- pybind11 Documentation: https://pybind11.readthedocs.io/
- デュエル・マスターズ公式ルール: `docs/DM_Official_Rules.md`

---

**最終更新**: 2026年1月9日  
**次回レビュー予定**: 2026年1月16日（Week 1完了時）  
**ドキュメント管理者**: Development Team
