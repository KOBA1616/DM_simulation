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
