# 要件定義書 00: ステータスと要件の概要

## 1. プロジェクト概要
**目標:** C++エンジンとPythonバインディングを使用したデュエル・マスターズAI/エージェントシステムの開発。
**現在のフェーズ:** フェーズ5 (汎用エンジン機能と特徴量抽出)
**焦点:** 堅牢性、検証可能性、リスク管理。特に、エンジンの「原子アクション分解」を行い、複雑な効果処理をシンプルかつデータ駆動で扱えるようにリファクタリングする。

## 2. 現在のステータス
*   **エンジン:** `GameState`、`ActionGenerator`、`EffectResolver`を備えたC++20コア。汎用的な効果解決（Effect Buffer, Stack Zone）を実装済み。
*   **進捗:** 基本的な対戦、コスト軽減、ハイパーエナジー、革命チェンジの実装が完了。
*   **Pythonバインディング:** `pybind11`統合 (`dm_ai_module`)。
*   **AI:** AlphaZeroスタイルのMCTS (`ParallelRunner`) + PyTorch学習 (`train_simple.py`)。
*   **GUI:** Python `tkinter` ベース (`app.py`) および `PyQt6` ベース (`card_editor.py`)。
*   **データ:** JSONベースのカード定義 (`data/cards.json`)。

## 3. 現在の開発段階 (Current Development Stage)
**ステータス:** **原子アクション分解によるエンジンリファクタリング (PLAY_CARD処理の3段階分解 - 完了)**

これまでの開発で、基本的なゲームルールといくつかの複雑なキーワード能力（革命チェンジ等）を実装しました。
現在は、エンジン内部の処理を「原子アクション（Atomic Actions）」に分解・再構築するフェーズにあります。
`PLAY_CARD` アクションを `DECLARE_PLAY`（宣言・スタック移動）, `PAY_COST`（コスト支払い）, `RESOLVE_PLAY`（解決） の3つの原子アクションに分解する実装が完了しました。

## 4. アクティブな要件とタスク (Uncompleted Tasks)

### 4.1. エンジンリファクタリング：原子アクション分解
以下の提案に基づき、エンジンのコアアクション処理を分解・再定義します。これにより、コードの重複を排除し、新しいメカニクス（G・ゼロ、シールド焼却など）の実装を容易にします。

#### 1. PLAY_CARD 処理の3段階分解 (完了)
現在の `resolve_play_card` は「宣言」「コスト支払い」「解決」を一括で行っていますが、これを以下のアクションに分解しました。
*   **DECLARE_PLAY:** カードをソース領域（手札、墓地など）から「スタック領域」へ移動させるアクション。
*   **PAY_COST:** マナコストを計算し、マナをタップするアクション。支払いに失敗した場合は失敗を返す。
*   **RESOLVE_PLAY:** スタックにあるカードを適切なゾーン（バトルゾーンまたは墓地）に移動させ、ON_PLAY（CIP）効果を誘発させるアクション。
    *   *メリット:* G・ゼロ/踏み倒し（PAY_COSTスキップ）、墓地詠唱（ソース変更）、シールドトリガーの共通化が可能になります。

#### 2. execute_battle の分解（バトル解決とブレイクの分離）
現在の `execute_battle` 関数（勝敗判定とブレイク処理の混在）を分解します。
*   **RESOLVE_BATTLE:** クリーチャー同士のパワーを比較し、敗北した方を破壊するアクション。
*   **BREAK_SHIELD:** （攻撃成功時）シールドを1枚指定して手札に加え、S・トリガー判定を行うアクション。W・ブレイカー等はこれを複数回生成します。
    *   *メリット:* シールド焼却（`BURN_SHIELD`アクションへの置換）などの実装が容易になります。

#### 3. マナチャージの汎用化
*   **MANA_CHARGE → MOVE_CARD:** マナチャージを汎用的な `MOVE_CARD` (Destination: MANA_ZONE) に統合します。

#### 4. 汎用アクションの完全定義
*   **A. MOVE_CARD (完全共通化):**
    *   アクション: `MOVE_CARD(source_zone, dest_zone, card_instance_id)`
    *   破壊、バウンス、ドロー、シールド化を全て統合し、「ゾーンを離れた時/置かれた時」のフック処理を一元化します。
*   **B. TAP_CARD / UNTAP_CARD (分離):**
    *   アクション: `TAP_CARD(target_instance_id)`
    *   攻撃、ブロック、支払いに含まれるタップ処理を分離し、効果によるタップ/アンタップを容易にします。
*   **C. SHUFFLE_DECK:**
    *   サーチや回復効果のために単独アクション化します。
*   **D. BATTLE_CREATURES:**
    *   ブレイク要素を排除した純粋なバトル処理として定義します。

#### 5. その他機能拡張のためのアクション
*   **APPLY_MODIFIER (継続的効果):**
    *   アクション: `APPLY_MODIFIER(modifier_def, duration, target_filter)`
    *   コスト軽減やパワー修正を履歴に残るアクションとして定義し、AIが理由を追跡可能にします。
*   **REVEAL_CARDS (公開):**
    *   アクション: `REVEAL_CARDS(zone, indices, visibility_scope)`
    *   「見る」処理を明確化し、AIの観測情報（Observation）更新ロジックと連動させます。
*   **REGISTER_DELAYED_EFFECT (遅延誘発):**
    *   アクション: `REGISTER_DELAYED_EFFECT(trigger_condition, effect_def)`
    *   「ターン終了時」などの効果予約をデータ駆動（JSON）で行えるようにします。
*   **RESET_INSTANCE (初期化):**
    *   アクション: `RESET_INSTANCE(instance_id)`
    *   ゾーン移動時の召喚酔いや修正値のリセットを明示的なアクションとして分離します。

### 4.2. GUI & カード効果の拡張
*   **カード作成補助 (日本語化 & 視覚化):**
    *   **日本語化:** GUI上のキーワード能力や汎用能力の選択を日本語で行えるようにする。
    *   **視覚的ビルダ:** トリガー条件、対象ゾーン、処理（見る、表向きにする、手札に戻す、参照など）、対象数、対象条件、コスト軽減などを視覚的に選択・組み合わせて複雑な効果を実装できる機能を追加する。

### 4.3. 将来の機能バックログ (ユーザー要望)
*   **システム & エンジン:**
    *   ドロー監視、墓地ロジックの拡張。
*   **カードメカニクス:**
    *   ジャストダイバー、代替コスト、メテオバーン、NEO進化、ニンジャ・ストライク、ブロック不可、全体除去、攻撃制限、アンチチート、マナ回収、リアニメイト、モード効果。

### 4.4. 実装計画：メタカウンターとエンジンコア拡張

#### 1. コア型定義とデータ構造の整備
- **Action と Types の更新**
    - `src/core/types.hpp` に `SpawnSource` enum を追加します。
        - `HAND_SUMMON` (手札からの通常召喚, G・ゼロ)
        - `EFFECT_SUMMON` (S・トリガー, メクレイド, コスト踏み倒し召喚)
        - `EFFECT_PUT` (リアニメイト, 踏み倒し出し)
    - `src/core/types.hpp` に新しい `EffectType` として `INTERNAL_PLAY`, `META_COUNTER` を追加します。
    - `src/core/action.hpp` に `ActionType::PLAY_CARD_INTERNAL` を追加し、`Action` 構造体に `spawn_source` フィールド（または既存フィールドの流用）を追加します。
- **カード定義の更新**
    - `src/core/card_def.hpp` の `CardKeywords` に `bool meta_counter_play` を追加します。
    - `src/engine/card_system/json_loader.cpp` を更新し、JSON からこのキーワードをパースできるようにします。
- **ゲーム状態の更新**
    - `src/core/game_state.hpp` に `struct TurnStats` を定義し、メンバとして `bool played_without_mana` を持たせます。
    - `GameState` クラスに `TurnStats turn_stats` を追加します。

#### 2. ターン統計機能の実装 (マナ踏み倒し検知)
- **リセット処理**
    - `src/engine/flow/phase_manager.cpp` の `start_turn` メソッド内で、`turn_stats` をリセットする処理を追加します。
- **フラグ更新処理**
    - `src/engine/mana/mana_system.hpp` (または `cpp`) を修正し、カードプレイ時に「支払われたマナ（タップされたマナ）」が0枚であるかを確認します。
    - 0枚の場合、`game_state.turn_stats.played_without_mana` を `true` に設定します（コスト軽減で1マナ払った場合は除外）。

#### 3. エンジンフローのリファクタリング (スタックとゲートキーパー)
- **EffectResolver の拡張**
    - `src/engine/effects/effect_resolver.cpp` の `resolve_play_from_stack` を修正し、`SpawnSource` 引数を受け取れるようにします。
    - **ゲートキーパー (Gatekeeper) ロジックの実装**:
        - カードがバトルゾーンに出る直前に、「移動先決定ロジック」を挟みます。
        - ここで将来的に実装されるメタカード（「マナよりコストが大きいならマナ送り」等）の判定フックを作成します。
        - デフォルトでは `Battle Zone` を返しますが、条件によって `Mana Zone`, `Deck Bottom`, `Graveyard` に変更できるようにします。
    - `ActionType::PLAY_CARD_INTERNAL` を処理するケースを追加し、`resolve_play_from_stack` へ委譲します。
- **ActionGenerator の更新**
    - `src/engine/action_gen/action_generator.cpp` を更新し、`EffectType::INTERNAL_PLAY` や `META_COUNTER` が `pending_effects` にある場合、`PLAY_CARD_INTERNAL` アクションを生成するようにします。

#### 4. 既存メカニクスのスタック移行
- **直接 `push_back` の廃止**
    - 以下の箇所で、直接バトルゾーンに追加している処理を廃止し、代わりに `PendingEffect` (Type: `INTERNAL_PLAY`) を積む形に変更します。
        - `src/engine/card_system/generic_card_system.cpp`: メクレイド (MEKRAID)、バッファからのプレイ
        - `src/engine/effects/effect_resolver.cpp`: S・トリガー (SHIELD_TRIGGER) の解決処理
        - その他のリアニメイト処理

#### 5. カウンター踏み倒し機能の実装
- **ターン終了時のチェック**
    - `src/engine/flow/phase_manager.cpp` の `next_phase` (END_OF_TURN 処理、またはターン切り替え前) に以下のロジックを追加します。
        1. `game_state.turn_stats.played_without_mana` が `true` かチェック。
        2. 自分の手札をスキャンし、`meta_counter_play` を持つカードを探す。
        3. 該当カードがある場合、**自分のバトルゾーン**に同名のカードが存在しないか確認する。
        4. 条件を満たすカードがあれば、`PendingEffect` (Type: `META_COUNTER`) を積む。
- **解決フロー**
    - 積まれた `META_COUNTER` は `ActionGenerator` によってアクション化され、スタック経由（`PLAY_CARD_INTERNAL`）で解決されます（召喚扱い `SpawnSource::EFFECT_SUMMON`）。

#### 6. 検証
- **テスト作成**
    - `tests/test_meta_counter.py` を作成し、以下を検証します。
        - マナなしプレイ時のフラグ検知。
        - ターン終了時のカウンター発動（条件合致時）。
        - 同名カードがある場合の発動阻止。
        - スタック経由でのカードプレイ（ゲートキーパー）が正常に動作すること。
- **Pre-commit**
    - 全テストの通過を確認し、コードフォーマット等を整えます。

## 5. 既知の問題 / リスク
*   **複雑な効果:** 複数ステップの効果 (探索、シールドトリガーの選択) はC++での堅牢な処理が必要です。
*   **メモリ使用量:** `verify_performance.py` でのシミュレーション回数が多いと、メモリアロケーションエラー (`std::bad_alloc`) が発生する可能性があります。
*   **データの一貫性:** `data/cards.csv` はレガシーであり、`data/cards.json` を使用する必要があります。

## 6. 今後の実装計画 (Future Implementation Plan)

### [PLAN-003] Transformerアーキテクチャの実装
*   **概要:** 現在のCNN/ResNetモデルから、Transformer (Self-Attention) モデルへの移行。
*   **目的:** 盤面の複雑な相互作用を捉え、より戦略的な深さを持つAI「自筆進化エコシステム」の基盤とする。

### [PLAN-004] PBT (Population Based Training) 基盤の構築
*   **概要:** 数千〜数万試合規模の並列対戦を実行し、エージェントを進化させるパイプラインの構築。
*   **目的:** メタゲームの変遷に適応する堅牢なAIを作成する。

## 7. 開発規約 (Development Conventions)
*   **コミットメッセージ:** 日本語で記述する。 (`[Type] Subject`)
