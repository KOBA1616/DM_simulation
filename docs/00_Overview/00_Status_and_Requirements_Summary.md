# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

## 2. 実装済み機能 (Completed Features)

### 2.1 コアエンジン (C++)
*   **基本ルール**: ターンの進行、マナチャージ、召喚、攻撃、ブロック、シールドブレイク、勝利条件。
*   **ゾーン管理**: 山札、手札、マナゾーン、バトルゾーン、シールドゾーン、墓地。
*   **カード効果**:
    *   `dm::engine::GenericCardSystem` によるJSONベースの効果処理。
    *   Trigger: ON_PLAY, ON_ATTACK, ON_DESTROY, S_TRIGGER, PASSIVE_CONST (Speed Attacker, Blocker, etc.), ON_ATTACK_FROM_HAND (Revolution Change).
    *   Actions: DRAW, ADD_MANA, DESTROY, TAP, UNTAP, RETURN_TO_HAND, SEARCH_DECK, SHUFFLE_DECK, ADD_SHIELD, SEND_SHIELD_TO_GRAVE, MEKRAID, REVOLUTION_CHANGE.
*   **JSON読み込み**: `dm::engine::JsonLoader` によるカード定義のロード。
*   **シミュレーション**: `dm::ai::MCTS` および `dm::ai::ParallelRunner` による並列モンテカルロ木探索。

### 2.2 AI & 学習 (Python/C++)
*   **モデル**: ResNetベースのニューラルネットワーク (`AlphaZeroNetwork` in PyTorch).
*   **推論**: C++からのバッチ推論コールバック (`register_batch_inference_numpy`)。
*   **データ収集**: 自己対戦による学習データ生成 (`collect_training_data.py`).
*   **学習ループ**: `train_simple.py` によるモデル更新。

### 2.3 GUI (Python/PyQt6)
*   **カードエディタ**: カードデータの作成・編集 (JSON形式)。日本語対応。
*   **シミュレーション対話**: 対戦の観戦やデバッグ。

## 3. 次のステップの要件 (Next Requirements)

### 3.1 テストコードの整理と動作確認
*   `tests/` ディレクトリ内のアドホックなテストを `python/tests/` へ統合。
*   原子アクション (Draw, Mana Charge, Tap, Break Shield, Move Card) の動作を検証する `python/tests/test_atomic_actions.py` の作成と維持。
*   **Status**: 完了 (2025/XX/XX)

### 3.2 GUIの日本語化拡充
*   `python/gui/card_editor.py` および `localization.py` を更新し、英語のハードコードを排除して日本語表示に対応する。
*   **Status**: 完了 (2025/XX/XX)

### 3.3 PythonコードのC++移行 (Migration Candidates)

パフォーマンス向上とロジックの堅牢化のため、以下のPython実装部分をC++へ移行することを計画しています。

1.  **シナリオ実行 (`ScenarioRunner`)**
    *   現状: `python/training/scenario_runner.py` でPython側でゲームループを回している。
    *   移行案: `src/core/scenario_config.hpp` は既にC++にあるため、`GameInstance::run_scenario(ScenarioConfig)` のようなメソッドをC++側に実装し、Pythonからは呼び出すだけにする。
    *   メリット: Python <-> C++ のオーバーヘッド削減、高速化。

2.  **デッキ進化/検証 (`DeckEvolution` / `VerifyPerformance`)**
    *   現状: `verify_deck_evolution.py` や `verify_performance.py` の一部ロジックがPython。
    *   移行案: 進化ロジック (遺伝的アルゴリズムの選択・交叉など) はPythonでも良いが、評価のための対戦実行ループは完全に `ParallelRunner` (C++) に任せる。
    *   補足: 既に `ParallelRunner` を使用しているが、セットアップや結果集計をよりC++側へ寄せ、Pythonは設定と起動のみにする。

3.  **複雑なカード効果のPython側ロジック (もしあれば)**
    *   現状: ほぼ全てのカード効果は `GenericCardSystem` (C++) に移行済み。
    *   確認事項: Python側で `register_card_functions` 等を使って実装されているレガシーな効果があれば、JSON定義 + C++実装へ完全移行する。

4.  **AI学習データ生成の制御**
    *   現状: `collect_training_data.py` が `dm_ai_module.DataCollector` を呼んでいるが、ループ制御の一部がPython。
    *   移行案: `DataCollector` が指定エピソード数を完遂するまでPythonに制御を戻さないようにする (現状も近い形だが、メモリ管理を厳密にするためC++側で完結させる)。

## 4. 今後のロードマップ (Roadmap)
*   **Phase 6**: サーチ、シールド操作の実装 (完了)。
*   **Phase 7**: 高度なギミック (超次元、GRなど) の検討。
*   **Phase 8**: AIモデルの高度化 (Transformerなど)。
