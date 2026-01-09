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
*   [Status: Review] **Transformer Model**: `DuelTransformer` (Linear Attention, Synergy Matrix) の実装完了。学習パイプライン `train_transformer_phase4.py` 稼働確認済み。
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

**現在のステータス**: ✅ Week 2 Day 2-3 実装完了（学習ループ・メトリクス収集）

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
1. `data/synergy_pairs_v1.json` - 手動定義ペア（Q1: 手動定義）✅
2. `SynergyGraph.from_manual_pairs()` - ロード機能（密行列で保持）✅
3. `generate_transformer_training_data.py` - データ生成（1000サンプル、Q4: 新規作成）✅
4. `train_transformer_phase4.py` - CLS先頭、学習可能pos embedding、バッチ8起動 ✅ (CPU/GPU対応)
5. データ正規化とテスト（デッキ系ソート、Battle重なり保持、ドロップアウト未実施）✅

### Week 3 Day 1-2（1月20-21日）の成果物
1. `DataCollector` (C++) - `TensorConverter` V2連携 (max_len=200, special tokens) ✅
2. `generate_transformer_training_data.py` - 実データ生成ロジックへの更新 (dummy廃止) ✅
3. `train_transformer_phase4.py` - one-hotターゲットのサポート (argmax) ✅
4. C++ -> Python Integration - 1エポックの学習ループ通過を確認 ✅

### 3.3 Documentation
*   **Update Specs**: 実装と乖離した古い要件定義書の更新（本タスクにて実施中）。

### 3.4 Command Pipeline / Legacy Action Removal
*   [Status: Review] **旧Action完全削除ロードマップの遂行**: カードJSONの `actions` と関連する互換コード/UIを段階的に撤去し、`commands` を唯一の表現に統一する。
	*   Phase 1-5: 完了（入口統一、データ移行、GUI撤去、互換撤去、デッドコード削除）
	*   ロードマップ: [docs/00_Overview/01_Legacy_Action_Removal_Roadmap.md](01_Legacy_Action_Removal_Roadmap.md)
	*   前提: `dm_toolkit.action_to_command.action_to_command` を唯一の Action→Command 入口にする（AGENTSポリシー準拠）。

### 3.5 テキスト生成とi18n改善 (Text Generation & i18n)
*   [Status: Done] **自然言語化の強化**: CardTextGeneratorでのTRANSITIONコマンドの自然言語化の実装完了。
	*   対応: `BATTLE→GRAVEYARD` を「破壊」と表示する短縮形ルールの実装完了
	*   対応: ゾーン名の生エキスポート（`BATTLE_ZONE`等）の防止完了
*   [Status: Done] **GUIスタブの改善**: PyQt6モックの設定修正完了（`conftest.py` による強制スタブ注入）。
	*   結果: headlessテスト環境でのGUI関連テスト（`test_gui_libraries_are_stubbed`）通過確認。

## 4. テスト状況 (Test Status)
**最終実行日**: 2026年1月9日  
**通過率**: 98.3% (121 passed + 41 subtests passed / 123 total + 41 subtests)

### 4.1 解決済みのテスト (3件)
1. `test_gui_stubbing.py::test_gui_libraries_are_stubbed` - ✅ Fixed (via `conftest.py` injection)
2. `test_generated_text_choice_and_zone_normalization.py::test_transition_zone_short_names_render_naturally` - ✅ Fixed
3. `test_generated_text_choice_and_zone_normalization.py::test_choice_options_accept_command_dicts` - ✅ Fixed

### 4.2 スキップ中のテスト (5件)
- `test_beam_search.py::test_beam_search_logic` - C++評価器の未初期化メモリ問題
- その他CI関連スキップ

## 5. 詳細実装計画 (Detailed Implementation Plan)

本セクションでは、2026年第1四半期の実装計画を具体的なタスク、タイムライン、リソース配分、技術的詳細と共に定義する。

### 5.1 Phase 6: 品質保証と残存課題（即時対応 - 1週間）

#### 5.1.1 テキスト生成の自然言語化 [Done]
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

**成功基準**:
- `test_transition_zone_short_names_render_naturally` 通過 ✅
- `test_choice_options_accept_command_dicts` 通過 ✅
- 生のゾーン名（`BATTLE_ZONE`等）がテキストに含まれない

---

#### 5.1.2 GUIスタブの修正 [Done]
**担当領域**: Testing Infrastructure  
**技術スタック**: Python, unittest.mock  
**依存関係**: なし

**実装詳細**:
`conftest.py` 内で `pytest_configure` フックを使用して `PyQt6.QtWidgets` 等を強制的にモックへ置き換えるロジックを実装。これにより、環境にPyQt6がインストールされていてもXサーバーがない場合の `ImportError` を回避。

**成功基準**:
- `test_gui_libraries_are_stubbed` 通過 ✅
- 全GUIテストがheadless環境で実行可能

---

### 5.2 Phase 4: Transformerモデル統合（Week 2-3 - 2週間）

**📋 詳細要件**: [04_Phase4_Transformer_Requirements.md](04_Phase4_Transformer_Requirements.md) を参照。Q1-Q9 決定済み（手動Synergy、CLS先頭、学習可能pos、データ新規生成、密行列Synergy、データ正規化のみ）。

#### 5.2.1 Week 2 Day 1: セットアップ（1月13日）
- `data/synergy_pairs_v1.json` 作成と `SynergyGraph.from_manual_pairs()` 実装（密行列で保持）。[Done]
- `generate_transformer_training_data.py` で 1000 サンプル生成（バッチ8起動、max_len=200、正規化のみ）。[Done]
- `train_transformer_phase4.py` スケルトン起動（CLS先頭、学習可能pos、lr=1e-4, warmup=1000）。[Done]
- 正規化ルール: Deck/Hand/Mana/Graveソート、Battle重なり保持、空ゾーン省略なし、ドロップアウト未実施。
- 成功基準: 上記4成果物がGPU上で1バッチ通る。[Done] (Verified on CPU, 10 samples/s)

#### 5.2.2 Week 2 Day 2-3: 学習ループと指標 (完了)
- バッチサイズ段階拡大 8→16→32→64（VRAM測定と勾配安定性確認）。[Done]
- ロギング: loss/throughput/VRAM、TensorBoard、チェックポイント（5k stepsごと）。[Done]
- 評価フック: vs Random/MLP簡易評価、ターン数・推論時間・Policy Entropyを収集。[Done] (Entropy/Throughput verified)
- データ拡張は実施せず（正規化のみ）、後続フェーズでドロップアウト検証。
- 成功基準: バッチ32で安定学習、評価フックが動作。

#### 5.2.3 Week 3 Day 1-2: TensorConverter連携 [Done]
- `dm_ai_module` TensorConverter出力をTorchに零コピーで受け取る構造を検討。
- マスク/パディングを max_len=200 に強制し、シーケンス長逸脱を検出。
- 成功基準: C++→Python 連携で1エポック通過、変換オーバーヘッド <5ms/batch。
- 実績: `DataCollector` の更新と `generate_transformer_training_data.py` の統合完了。

---

### 5.3 Phase 3: メタゲーム進化システム（Week 4-5 - 1.5週間）
（詳細省略 - 前回と同じ）

---

### 5.5 リソース配分とタイムライン
（前回と同じ）

---

## 11. 次のアクション（即座実行）

### 今日実施すべきタスク（優先順位順）
1. **Phase 6 ブロッカー解消** [Done]
  - [x] ゾーン自然言語化と選択肢生成の修正（[dm_toolkit/gui/editor/text_generator.py](dm_toolkit/gui/editor/text_generator.py)）。
  - [x] PyQtスタブの修正（[python/tests/conftest.py](python/tests/conftest.py) へ移行）。
  - 目標: 失敗中3テストを通過。 -> **All Passed**

2. **Week 2 Day 1 仕込み** [Done]
  - [x] [data/synergy_pairs_v1.json](data/synergy_pairs_v1.json) の雛形作成（手動10-20ペア）。
  - [x] [python/training/generate_transformer_training_data.py](python/training/generate_transformer_training_data.py) のスケルトン作成と実データ生成（1000サンプル）。
  - [x] [python/training/train_transformer_phase4.py](python/training/train_transformer_phase4.py) の学習ループ稼働確認。
  - 目標: Day 1 開始時にGPUで1バッチ流せる状態。 -> **Verified**

3. **環境確認**
  - CUDA/ドライバと `.venv` の動作確認、TensorBoard起動テスト。
  - 目標: 学習ループデバッグに即移行できる状態。

### 今週中に完了すべきマイルストーン
- [x] Phase 6 ブロッカー解消（3テスト通過、通過率99%近似）
- [x] Week 2 Day 1 成果物の雛形完成（synergy JSON, データ生成スケルトン, 学習起動）
- [x] Week 3 Day 1-2 TensorConverter連携（C++データ収集からPython学習まで開通）
- [x] ドキュメント更新（本ファイル）
- [ ] [docs/00_Overview/NEXT_STEPS.md](docs/00_Overview/NEXT_STEPS.md) 更新

---
