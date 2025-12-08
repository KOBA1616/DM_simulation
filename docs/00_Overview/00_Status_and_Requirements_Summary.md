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
    *   **スタックシステムの実装 (2025/03/XX)**: トリガー能力 (`ON_PLAY`など) の即時解決を廃止し、`PendingEffect` リスト（スタック）に一度積んでから、プレイヤー（またはAI）が順次解決する仕組みに変更しました。これにより、連鎖的な能力発動や、同時発動時の解決順序選択が可能になりました。
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

### 2.4 テスト状況と課題 (Testing Status & Identified Issues) (2025/03/XX 更新)

#### テスト実行結果
Pythonバインディング (`bindings.cpp`) の不整合を解消し、テストスイートの大部分が通過することを確認しました。
*   **バインディング修正**: `GameInstance`, `GameState` (current_phase, stack_zone, turn_stats), `CardInstance`, `ActionDef`, `FilterDef` の欠落していたメソッドやプロパティを追加。
*   **Enum定義**: `SpawnSource`, `DECLARE_PLAY`, `PAY_COST`, `RESOLVE_PLAY` などを追加。
*   **AI関連**: `POMDPInference`, `ParametricBelief` のスタブを追加。
*   **Numpy依存**: テストに必要な `numpy`, `torch` をインストール。

#### 修復済みのテスト (Resolved Issues)
*   **`python/tests/test_meta_counter.py` の修正**: `ActionGenerator` (PendingEffectStrategy) が生成する `PLAY_CARD_INTERNAL` アクションに `card_id` が正しく設定されておらず、テスト側のフィルタリングで除外されていました。C++側の `PendingEffectStrategy::generate` を修正し、`bindings.cpp` の `add_card_to_hand` 等で `card_owner_map` を正しく更新するようにしたことで、`game_state.get_card_instance` が正常に機能し、`card_id` が設定されるようになりました。
*   **`python/tests/test_engine_basics.py` のリファクタリング**: 非推奨の `GameInstance` 依存コードを最新の `GameState` と `PhaseManager` を使用する形に書き換え、最新のエンジンAPIとの互換性を確保しました。
*   **`python/tests/test_atomic_actions.py` の修正**: `DevTools.move_cards` の引数不足およびダミーカードの追加漏れによるテスト失敗を修正しました。

#### 特定された課題 (Remaining Issues)
*   `python/tests/test_just_diver.py` および `test_json_loader.py` の一部アサーション失敗は、テストデータまたはテストシナリオの修正が必要と考えられます。
    *   特に `test_just_diver_attack.py` は、ターン進行 (`turn_number`) と ジャストダイバーの解除タイミングの解釈に齟齬がある可能性があります。

これらの問題は次のフェーズで対処予定ですが、コアエンジンの主要機能（マナ、召喚、攻撃、シールドブレイク、効果解決）は安定しています。

## 3. 実装完了機能 (Phase 6 & 7 Implementation Summary)

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

### Phase 8 (New): スタックシステムとビルド安定化 (Stack System & Build Stabilization)

6.  **トリガースタック (Trigger Stack System) - 実装済み (2025/03/XX)**
    *   トリガー能力（CIP等）を即時実行せず、`PendingEffect` (Type: `TRIGGER_ABILITY`) として待機リストに追加。
    *   `EffectResolver` にて `RESOLVE_EFFECT` アクションを介して順次実行する仕組みを実装。
    *   `python/tests/test_trigger_stack.py` にて基本動作（トリガーの保留と解決）を検証済み。

7.  **ビルド環境の修復と安定化 (Build Fixes)**
    *   `CardDefinition` の文明定義（単数→複数）変更に伴うコンパイルエラー（`ManaSystem`, `JsonLoader`, `CsvLoader`, `TensorConverter`）を修正。
    *   Pythonバインディング (`bindings.cpp`) のインクルード漏れや型定義の不整合（`CardKeywords`, `Zone` Enum）を修正し、`dm_ai_module` の正常ビルドを回復。

## 4. 今後のロードマップ (Roadmap)
*   **スタックシステムの完全な統合**: 既存のAIモデルや戦略が、即時実行ではなくスタック解決アクションを適切に選択できるか、自己対戦を通して検証する。
*   **テスト環境の整備**: `test_just_diver.py` や `test_json_loader.py` の失敗原因を特定し、修正する。
*   **Phase 9**: AIモデルの高度化 (PPO + LSTM アーキテクチャ) および GUI連携の強化。

## 5. Phase 9 要件定義: AIモデルの高度化 (AI Model Refinement with PPO+LSTM)

「現代のゲームAIのデファクトスタンダード」である **PPO + LSTM (Actor-Critic)** 構成を採用し、ソリティア（大量行動・ループ）への対応と不完全情報下での手札予測能力を強化します。

### 5.1 アーキテクチャ概要
*   **基本構成**: Actor-Critic (PPO) + LSTM
    *   **Actor (Policy Head)**: 最善手（方策）を確率的に決定。
    *   **Critic (Value Head)**: 現在の盤面の勝率（価値）を評価し、危機察知センサーとして機能させる。
    *   **LSTM (Memory)**: 過去の盤面や行動の文脈（Hidden State）を記憶し、「相手の手札予測」や「ゲームの流れ」を推論する。
*   **入力層**: 盤面ベクトル + カード埋め込み (BERT等) + **墓地・マナのコンボパーツ密度カウンタ**。
*   **時系列処理**: 直近の行動すべてではなく、「過去数ターンの盤面スナップショット」＋「直近30手のアクション」程度に圧縮して入力。

### 5.2 学習戦略 (カリキュラム学習)

#### Step A: 模倣学習 (Imitation Learning / Behavior Cloning)
*   **目的**: ループやコンボの「型（定跡）」を覚えさせる。
*   **手法**: ルールベースのボット（スクリプト）または人間のプレイログを教師データとし、教師あり学習を行う。

#### Step B: 報酬シェイピング (Reward Shaping)
*   **目的**: 勝利報酬だけでは到達困難な複雑なループ手順を学習させる。
*   **手法**: 中間報酬を設定する。
    *   キーパーツが揃った (+0.1)
    *   マナ/墓地が特定枚数を超えた (+0.1)
    *   相手シールドを0にした (+0.5)
*   **ループ検知**: ゲームエンジン側でループ成立（無限行動）を検知し、強制勝利として学習を打ち切る判定を実装。

#### Step C: Action Masking (無効な手の排除)
*   **目的**: 探索空間の爆発を防ぐ。
*   **手法**: ルール上不可能な手（マナ不足、対象不在）をニューラルネットの出力段階でマスク（確率0）し、有効な手のみからサンプリングさせる。

### 5.3 防御と対戦 (Defense & Self-Play)
*   **危機察知**: Critic (Value Network) の急激な値の低下（＝相手がコンボ準備に入った）を「危機」として検知し、それを回復する行動（メタカード、ハンデス）を学習させる。
*   **MCTSのロールアウト強化**: ランダムではなくPPO Policyを用いたロールアウトを行い、ループデッキ相手でも「正しく負ける」シミュレーション精度を確保する。
*   **プール学習**: 「過去の自分」や「ループ特化スクリプト」を対戦相手としてプールし、多様な戦略への適応力を高める。
