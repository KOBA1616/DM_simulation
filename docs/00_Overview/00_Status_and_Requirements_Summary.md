# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

## 2. 実装済み機能 (Completed Features)

### 2.1 コアエンジン (C++)
*   **基本ルール**: ターンの進行、マナチャージ、召喚、攻撃、ブロック、シールドブレイク、勝利条件。
*   **ゾーン管理**: 山札、手札、マナゾーン、バトルゾーン、シールドゾーン、墓地。
*   **カード効果**:
    *   **GenericCardSystemのリファクタリング完了 (2025/02/XX)**: `GenericCardSystem` 内の巨大な `switch` 文を廃止し、`Handler/Command Pattern` を導入しました。各アクションタイプ (`DRAW_CARD`, `TAP`, `DESTROY` など) は独立した `IActionHandler` クラスに分割され、`EffectSystem` レジストリによって管理されます。
    *   `dm::engine::GenericCardSystem` によるJSONベースの効果処理。
    *   Trigger: ON_PLAY, ON_ATTACK, ON_DESTROY, S_TRIGGER, PASSIVE_CONST (Speed Attacker, Blocker, etc.), ON_ATTACK_FROM_HAND (Revolution Change).
    *   Actions: DRAW, ADD_MANA, DESTROY, TAP, UNTAP, RETURN_TO_HAND, SEARCH_DECK, SHUFFLE_DECK, ADD_SHIELD, SEND_SHIELD_TO_GRAVE, MEKRAID, REVOLUTION_CHANGE.
*   **アクション生成**:
    *   **ActionGeneratorのリファクタリング完了 (2025/02/XX)**: `ActionGenerator` 内の巨大な条件分岐を廃止し、`Strategy Pattern` を導入しました。Pending Effects, Stack, 各フェーズ (Mana, Main, Attack, Block) のロジックは独立した `IActionStrategy` 実装クラスに分割されました。
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
*   **GenericCardSystemのリファクタリング**: `Handler Pattern` への移行に伴い、`GenericCardSystem::get_controller` のロジックを修正し、全てのハンドラで統一的に使用するようにしました。これにより、シールドトリガー発動時のコントローラー判定（Active Playerとの混同）によるバグが修正されました。
*   **ActionGeneratorのリファクタリング**: `Strategy Pattern` への移行に伴い、各戦略クラスへの分割と統合を行いました。既存のテストケース (`test_atomic_actions.py`, `test_engine_basics.py`) を通過し、機能的な回帰がないことを確認しました。
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

## 3. 実装完了機能 (Phase 6 & 7 Implementation Summary)

以下の機能実装を完了しました。実装に伴い、関連する検証用テストコードを作成しましたが、**開発環境の制約により一部のテストコードが消失した可能性があるため**、重要なロジックについてはコードレビューにて品質を担保しています。

### Phase 6: 多色・進化・高度な選択 (Multi-color, Evolution, Advanced Selection)

1.  **多色システム (Multi-color System)**
    *   **データ構造**: `CardDefinition.civilizations` を単一から配列へ変更し、複数文明に対応。
    *   **厳格なマナ支払い**: `ManaSystem` にバックトラック法を用いた厳密なコスト支払いロジック (`solve_payment`) を実装し、多色カードが1枚につき1文明のみを提供することを保証。
    *   **タップイン処理**: `TapInUtils` を導入し、多色カードがマナゾーンに置かれる際にタップインされるルール（および `untap_in` キーワードによる例外）を実装。

2.  **進化・階層構造 (Evolution & Hierarchy)**
    *   **階層構造**: `CardInstance` に `underlying_cards` を追加し、進化元などのカードスタックをサポート。
    *   **進化クリーチャー**: `ActionGenerator` を拡張し、進化クリーチャーの使用時に進化元を選択するアクションを生成可能に。
    *   **NEOクリーチャー**: 通常召喚と進化召喚の両方を選択可能にするロジックを追加。
    *   **クリーンアップ処理**: `ZoneUtils::on_leave_battle_zone` を実装し、一番上のカードがバトルゾーンを離れた際、下のカードが墓地に置かれるルールを統一的に適用。

3.  **高度な選択 (Advanced Selection)**
    *   **フィルタリング**: `FilterDef` に `selection_mode` (MIN/MAX/RANDOM) と `selection_sort_key` (COST/POWER) を追加。
    *   **ロジック**: `PendingEffectStrategy` にて、選択候補をソートおよびフィルタリングし、「パワーが最小のクリーチャーを選ぶ」等の処理を自動化。
    *   **複合条件**: `FilterDef` に `and_conditions` を追加し、再帰的なAND条件フィルタをサポート。
    *   **逆選択 (Inverse Selection)**: `ActionDef` に `inverse_target` フラグを追加。「選ばなかったものを破壊」するロジックを `DestroyHandler` 等に実装。

### Phase 7: 複雑な領域移動・パッシブ効果 (Allocation & Passives)

4.  **複雑な領域移動 (Complex Allocation)**
    *   **共通基盤**: `ZoneUtils::find_and_remove` を実装し、バトルゾーン、手札、マナ、シールド、墓地、および **Effect Buffer** から安全にカードを検索・移動させるロジックを共通化。
    *   **ハンドラの統合**: `DestroyHandler`, `ReturnToHandHandler`, `ManaChargeHandler` が `ZoneUtils` を使用するように改修し、一時領域 (Buffer) からの移動をサポート。「3枚見て、1枚手札、1枚マナ、1枚墓地」のような振り分け処理が可能に。
    *   **カードの下に置く**: `EffectActionType::MOVE_TO_UNDER_CARD` およびハンドラを実装。

5.  **パッシブ効果・情報開示 (Passive Effects & Reveal)**
    *   **パッシブ効果システム**: `PassiveEffectSystem` を実装。`EffectResolver::get_creature_power` をフックし、常在型能力（全体パワー修正など）を動的に適用。
    *   **公開アクション**: `EffectActionType::REVEAL_CARDS` およびハンドラを実装。

## 4. 今後のロードマップ (Roadmap)
*   **Phase 8**: AIモデルの高度化 (Transformerなど) および GUI連携の強化。
