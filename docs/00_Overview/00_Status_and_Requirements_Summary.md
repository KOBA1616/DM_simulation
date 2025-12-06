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
*   **カードエディタ**: カードデータの作成・編集 (JSON形式)。日本語対応。Ver 2.0 (Logic Tree + Property Inspector) へ改修済み。
*   **シミュレーション対話**: 対戦の観戦やデバッグ。

### 2.4 テスト状況と課題 (Testing Status & Identified Issues) (2025/XX/XX 更新)

#### テスト実行結果
既存のPythonテストスイート (`python/tests/`) に対して網羅的な実行を行い、以下の修正を行いました。
*   **テストコードの修正**: C++バインディング (`dm_ai_module`) の最新仕様に合わせて `CardData` コンストラクタ呼び出し等を修正しました。
*   **非推奨コードの削除**: `ActionType.DIRECT_ATTACK` 等の削除された定数への参照を修正しました。

#### 特定された課題 (Engine Issues)
テスト実行により、C++エンジン側の以下の挙動に関する課題が特定されました。

1.  **Meta Counter (Internal Play) の挙動不整合**
    *   **現象**: `ActionType::PLAY_CARD_INTERNAL` が手札 (`HAND_SUMMON`) から生成された場合、`EffectResolver` がカードをスタックへ移動せず、手札に残ったまま解決しようとして失敗する。
    *   **原因**: `EffectResolver` の `resolve_play_from_stack` は、スタックまたはバッファ内のカードのみを検索対象としており、手札からの直接的な内部プレイを想定していない（あるいは移動ロジックが欠落している）。
    *   **Status**: **修正完了 (2025/XX/XX)** - `ActionGenerator` にコントローラー情報の伝播を追加し、`EffectResolver` に `PLAY_CARD_INTERNAL` 実行後のPending Effect削除処理を追加しました。`python/tests/test_meta_counter.py` を通過することを確認済み。

2.  **Ninja Strike (Reaction System) の不発**
    *   **現象**: Ninja Strikeの条件を満たす状況下でも、リアクションウィンドウ（`PendingEffect`）が生成されるものの、`DECLARE_REACTION` アクションが生成されない。
    *   **原因**: エンジンイベント `"ON_ATTACK"` と JSON定義 `"ON_BLOCK_OR_ATTACK"` の文字列完全一致比較により、条件不一致と判定されていた。
    *   **Status**: **修正完了 (2025/02/XX)** - `ReactionSystem` および `ActionGenerator` にて、`"ON_BLOCK_OR_ATTACK"` が `"ON_ATTACK"`/`"ON_BLOCK"` イベントにもマッチするように緩和処理を追加。`python/tests/test_ninja_strike.py` を通過。

3.  **Just Diver (Play Action) の生成失敗**
    *   **現象**: Just Diverを持つクリーチャーのプレイアクションは生成されるが、プレイ後のターンでも対戦相手が対象に取れてしまう。
    *   **原因**: クリーチャーがバトルゾーンに出る際、`CardInstance.turn_played` プロパティが設定されておらず、期間判定（「このターン」）が正しく機能していなかった。
    *   **Status**: **修正完了 (2025/02/XX)** - `EffectResolver::resolve_play_from_stack` にて `turn_played` を設定するよう修正。`TargetUtils` のターン判定ロジックを修正。`python/tests/test_just_diver.py` を通過。

4.  **Pythonバインディングの制約**
    *   `std::vector` を返すプロパティ（`mana_zone` 等）はPython側ではコピーとなるため、要素への代入（`is_tapped = False`）がC++側に反映されない。テストコードでは `add_card_to_mana` 等の専用ヘルパーを使用する必要がある。

## 3. 次のステップの要件 (Next Requirements)

### 3.1 カードエディタ ユーザビリティ改善 (GUI UX Improvements) (2025/XX/XX 追加)

**目的**: 汎用的なGUIにより生じている「入力項目の分かりにくさ」と「変数管理の手間」を解消する。

**1. コンテキストに応じたUIの動的変更**
*   **ラベルの動的変更**: `Action Type` の選択に応じて、汎用的な `Value 1`, `Value 2` ラベルを具体的な意味（例: `Value 1` -> "枚数", `Value 2` -> "持続ターン"）に変更する。
*   **不要項目の非表示/無効化**: 選択中の `Action Type` で使用しないフィールド（例: ドロー時の `String Value` や `Filter`）をグレーアウトまたは非表示にし、入力すべき項目を明確化する。
*   **ツールチップの充実**: 各入力項目にマウスホバー時、そのアクションタイプでの具体的な使用方法を表示する。

**2. 変数連携の自動化と可視化 (Automated Variable Linking)**
*   **変数名の自動管理 (Implicit Variable Naming)**:
    *   ユーザーが手動で `Output Key` (変数名) を入力する手間を排除する。
    *   値を生成するアクション（`GET_GAME_STAT`等）を選択した際、システムが内部的に一意なキー（または自動生成された名前）を割り当てる。
*   **入力ソースの明示的選択**:
    *   後続のアクションの `Input Key` 選択肢において、単なる変数名ではなく「ステップ1: 文明数取得の結果」のように、どのアクションの出力であるかを分かりやすく表示する。
    *   これにより、ユーザーは「変数名」を意識することなく、ロジックの流れ（フロー）として値を参照できる。

**3. テンプレート機能 (Templates)**
*   よく使われるアクションの組み合わせ（例: 「マナ武装」「〜枚引いて〜枚捨てる」）をプリセットとして提供し、ワンクリックで展開できる機能を検討する。

**Status**: 実装予定 (Planned)
*   **1. コンテキストに応じたUIの動的変更**: 実装完了 (2025/XX/XX)
*   **2. 変数連携の自動化と可視化**: 実装完了 (2025/XX/XX)

## 4. 今後のロードマップ (Roadmap)
*   **Phase 6**: サーチ、シールド操作の実装 (完了)。
*   **Phase 7**: 高度なギミック (超次元、GRなど) の検討。
*   **Phase 8**: AIモデルの高度化 (Transformerなど)。
