# 完了タスクアーカイブ (Completed Tasks Archive)

このファイルは、`00_Status_and_Requirements_Summary.md` から完了したタスクを移動したアーカイブです。

## 完了済み機能 (Archived)

### 2.1. 実装済み機能 (過去の完了分)
*   **スピードアタッカー / 進化ロジック:** `スピードアタッカー`や`進化`を持つクリーチャーが（召喚酔いを無視して）即座に攻撃できるようにエンジンを修正しました。`verify_lethal_puzzle.py`で検証済み。
*   **バウンス (手札に戻す):** C++エンジンおよびGUIテンプレートで`RETURN_TO_HAND`アクションタイプを実装しました。
*   **デッキ探索/確認:** `SEARCH_DECK_BOTTOM` (上からN枚見て、選択したものを手札に加え、残りを一番下に戻す) を実装しました。
*   **メクレイド:** `MEKRAID` (上から3枚見て、条件に合うものを出し、残りを一番下に戻す) を実装しました。
*   **GUIエディタ:** `card_editor.py`を更新し、上記効果のテンプレートを追加しました。
*   **一貫性のあるJSONデータ:** ユニットテストが通過するように、必須のテストカード (デーモン・ハンド、スパイラル・ゲート) を`data/cards.json`に追加しました。
*   **EffectResolverのリファクタリング:** 呪文およびシールドトリガーの効果解決において`GenericCardSystem`を優先的に使用するように`EffectResolver`をリファクタリングし、コードの重複を削減しました。
*   **汎用ターゲット選択の修正:** `ActionGenerator`が`SELECT_TARGET`アクションを正しく処理し、ターゲット選択後に`RESOLVE_EFFECT`アクションを生成するように修正しました。また、`PendingEffect`がアクション定義を保持するように改善しました。
*   **Effect Bufferと汎用原子アクション:** [PLAN-002] Section 2完了。`LOOK_TO_BUFFER`, `SELECT_FROM_BUFFER`, `PLAY_FROM_BUFFER`, `MOVE_BUFFER_TO_ZONE` を実装し、中断・再開可能な選択ロジックを確立しました。`tests/test_effect_buffer.py` で動作検証済み。
*   **スタックゾーンとコスト計算:** [PLAN-002] Section 3完了。カードプレイ時に一時的に`stack_zone`を経由し、`get_projected_cost`および`auto_tap_mana`（軽減適用済み）を使用してコストを支払うフローを確立しました。`tests/test_engine_basics.py` および `tests/test_cost_modifier.py` で検証済み。

### 3.1. 汎用エンジン機能 (実装済み/テスト済み)
*   **汎用ターゲット選択:**
    *   **ステータス:** 実装済み。`tests/manual/test_generic_targeting.py` で検証完了。`SELECT_TARGET` -> `RESOLVE_EFFECT` のフローが正常に動作します。
*   **汎用アクション:**
    *   **ステータス:** `TAP`、`UNTAP`、`DESTROY`、`RETURN_TO_HAND` が `GenericCardSystem` に実装済み。
*   **Pythonバインディング:** `FilterDef` と `ActionDef` をサポートするように更新されました。
*   **ハイパーエナジー:**
    *   **ステータス:** 実装・検証完了。`ActionType.PLAY_CARD` の `target_player=254` を用いたタップ処理フローを確立。`tests/test_hyper_energy.py` にて検証済み。

### 3.4. 汎用エンジン機能 (完了分)
*   **一般化されたターゲット選択:**
    *   **完了:** `ActionType::SELECT_TARGET` フローの実装と修正が完了しました。
*   **一般化されたタップ/アンタップ:**
    *   **完了:** `TAP` / `UNTAP` 効果アクションタイプの実装完了。
*   **コスト軽減システム:**
    *   **完了:** `CostModifier` 構造体と `ManaSystem` への統合が完了しました。`tests/test_cost_modifier.py` で検証済み。

### [PLAN-001] コスト軽減システムの実装
*   **概要:** `CostModifier` 構造体と `ManaSystem` への統合。
*   **ステータス:** 完了 (Verified).

### [PLAN-002] スマートテスト＆エンジンリファクタリング要件 (完了分)
#### 1. データ収集 (Data Collection) の完全C++化
*   **HeuristicAgentのC++移植:** (完了)
*   **DataCollectorクラスの作成:** (完了)
*   **Pythonバインディングの更新:** (完了)

#### 2. エフェクトバッファによる汎用アクションシステム (完了)
*   **ステータス:** 完了 (Verified via `tests/test_effect_buffer.py`)

#### 3. スタック（宣言）ゾーンとコスト計算の再構築
*   **ステータス:** 完了 (Implemented & Verified)

### [PLAN-003] 革命チェンジ (Revolution Change)
*   **概要:** 革命チェンジのエンジン実装とGUI統合。
*   **ステータス:** 完了 (Verified via `tests/test_revolution_change.py`)
    *   専用アクション `REVOLUTION_CHANGE` とトリガー `ON_ATTACK_FROM_HAND` の実装。
    *   `FilterDef` を使用した革命チェンジ条件（文明、種族、コスト）の定義。
    *   GUIエディタでの設定機能。

### [Refactoring Phase] 原子アクション分解とメタカウンター (Atomic Actions & Meta Counter)

#### バトル処理の原子アクション化 - 実装完了
エンジンのリファクタリングフェーズにおいて、バトル解決処理の分解（`execute_battle` の `RESOLVE_BATTLE` と `BREAK_SHIELD` への分離）が完了しました。
これにより、ブロック後の処理やシールドブレイクが独立したアクションとして `ActionGenerator` によって生成され、`PendingEffect` を経由して制御されるようになりました。
これはシールド焼却や、より複雑な攻撃解決ロジック（置換効果など）を実装するための基盤となります。

#### PLAY_CARD 処理の3段階分解 (完了)
現在の `resolve_play_card` は「宣言」「コスト支払い」「解決」を一括で行っていますが、これを以下のアクションに分解しました。
*   **DECLARE_PLAY:** カードをソース領域（手札、墓地など）から「スタック領域」へ移動させるアクション。
*   **PAY_COST:** マナコストを計算し、マナをタップするアクション。支払いに失敗した場合は失敗を返す。
*   **RESOLVE_PLAY:** スタックにあるカードを適切なゾーン（バトルゾーンまたは墓地）に移動させ、ON_PLAY（CIP）効果を誘発させるアクション。
    *   *メリット:* G・ゼロ/踏み倒し（PAY_COSTスキップ）、墓地詠唱（ソース変更）、シールドトリガーの共通化が可能になります。

#### execute_battle の分解（バトル解決とブレイクの分離） (完了)
現在の `execute_battle` 関数（勝敗判定とブレイク処理の混在）を分解しました。
*   **RESOLVE_BATTLE:** クリーチャー同士のパワーを比較し、敗北した方を破壊するアクション。
*   **BREAK_SHIELD:** （攻撃成功時）シールドを1枚指定して手札に加え、S・トリガー判定を行うアクション。W・ブレイカー等はこれを複数回生成します。
    *   *実装詳細:* ブロックフェーズの終了時 (`PASS`) に、状況に応じて `RESOLVE_BATTLE` または `BREAK_SHIELD` の `PendingEffect` を発行し、それを `ActionGenerator` がアクションに変換して処理するフローに変更しました。

#### マナチャージの汎用化 (完了)
*   **MANA_CHARGE → MOVE_CARD:** マナチャージを汎用的な `MOVE_CARD` (Destination: MANA_ZONE) に統合します。
    *   *実装詳細:* `ActionType` に `MOVE_CARD` を追加し、マナフェーズでのアクション生成を `MOVE_CARD` に移行しました。`EffectResolver` での処理も対応済みです。（後方互換性のため `MANA_CHARGE` Enumは維持していますが、生成ロジックは更新されています）

#### コア型定義とデータ構造の整備 (完了)
- **Action と Types の更新** (完了)
    - `src/core/types.hpp` に `SpawnSource` enum を追加します。
    - `src/core/types.hpp` に新しい `EffectType` として `INTERNAL_PLAY`, `META_COUNTER`, `RESOLVE_BATTLE`, `BREAK_SHIELD` を追加します。
    - `src/core/action.hpp` に `ActionType::PLAY_CARD_INTERNAL`, `RESOLVE_BATTLE`, `BREAK_SHIELD` を追加します。
- **カード定義の更新** (完了)
    - `src/core/card_def.hpp` の `CardKeywords` に `bool meta_counter_play` を追加します。
- **ゲーム状態の更新** (完了)
    - `src/core/game_state.hpp` に `struct TurnStats` を定義し、メンバとして `bool played_without_mana` を持たせます。

#### ターン統計機能の実装 (マナ踏み倒し検知) (完了)
- **リセット処理** (完了)
    - `src/engine/flow/phase_manager.cpp` の `start_turn` メソッド内で、`turn_stats` をリセットする処理を追加します。
- **フラグ更新処理** (完了)
    - `src/engine/mana/mana_system.hpp` (または `cpp`) を修正し、カードプレイ時に「支払われたマナ（タップされたマナ）」が0枚であるかを確認します。
