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
    *   移行案: `ScenarioExecutor` (C++) を実装し、対戦ループをC++側で完結させる。
    *   **Status**: 完了 (2025/12/XX) - `ScenarioExecutor` クラス実装済み、`ScenarioRunner` から利用。

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

### 3.4 変数連携システム (Variable Linking System)

アクション間で値を渡すための「実行コンテキスト (execution_context)」を強化し、動的な数値参照を実現します。

*   **コンテキストの実装**: `PendingEffect` および `GenericCardSystem` 内で `std::map<std::string, int> execution_context` を保持・伝播させる。
*   **新規アクション**:
    *   `COUNT_CARDS`: 指定ゾーン（BATTLE_ZONE, GRAVEYARD等）の条件に合うカード数をカウントし、変数に保存する。
    *   `GET_GAME_STAT`: マナゾーンの文明数 (`MANA_CIVILIZATION_COUNT`) 等の統計値を取得し、変数に保存する。
    *   `SEND_TO_DECK_BOTTOM`: 手札等から選択したカードを山札の下に送る。
*   **既存アクションの拡張**:
    *   `DRAW_CARD` 等で `input_value_key` が設定されている場合、固定値ではなくコンテキスト内の変数値を使用する。
*   **目的**: 「自分のクリーチャーの数だけドローする」「マナゾーンの文明数分ドローして戻す」といった複雑な効果をJSON定義のみで実現可能にする。
*   **GUI対応**: カードエディタ (V2) にて、アクション間の変数リンク（Input/Output Key）設定UIを実装済み。
*   **Status**: 完了 (2025/XX/XX)

### 3.5 カードエディタの刷新 (Card Editor V2)
*   **アーキテクチャ変更**: タブ切り替え式から、左ペイン（ツリービュー）と右ペイン（プロパティインスペクタ）の分割画面 (QSplitter) へ刷新。
*   **階層構造の可視化**: Card -> Keywords/Effect -> Action の階層をツリーで表現し、ドラッグ＆ドロップによる並べ替えに対応。
*   **詳細エディタ**: アクションタイプごとの動的な入力フォーム切り替え、およびフィルタ（文明、種族、コスト、パワー範囲、タップ/ブロッカー/進化フラグ）の詳細設定を実装。
*   **変数リンク機能**: ツリー構造を解析し、先行するアクションの `Output Key` を自動的にリストアップして `Input Key` の候補として提示する機能を実装。
*   **Status**: 完了 (2025/XX/XX)

## 4. 今後のロードマップ (Roadmap)
*   **Phase 6**: サーチ、シールド操作の実装 (完了)。
*   **Phase 7**: 高度なギミック (超次元、GRなど) の検討。
*   **Phase 8**: AIモデルの高度化 (Transformerなど)。
