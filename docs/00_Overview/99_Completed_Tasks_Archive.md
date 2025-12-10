# 完了したタスクのアーカイブ (Completed Tasks Archive)

このドキュメントは `00_Status_and_Requirements_Summary.md` から移動された、完了済み要件の履歴です。

---

## Phase 0.5-1.0: Foundation & Core Features (Archived)

### Engine & AI Core
*   **MCTS & AlphaZero**: 基礎実装完了。
*   **並列実行 (`ParallelRunner`)**: 実装完了。
*   **Pythonバインディング**: `dm_ai_module` として実装完了。

---

## Phase 1.5: カードエディタ UI/UX 改善 (Completed)

カードデータの入力効率と表現力を向上させ、複雑なカード（多色、ツインパクトなど）の実装を可能にするための改修。

1.  **多色（マルチカラー）カード実装支援**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `CivilizationSelector` ウィジェットを作成し、`CardEditForm` に統合。
        *   単一選択ではなく、複数の文明（Fire, Water, etc.）をチェックボックス形式で選択可能にし、`civilizations` リストとしてJSONに保存する機能を実装。

2.  **ツインパクトカードの視覚的統合**
    *   **完了日**: 2024/05/25
    *   **実装内容**:
        *   `CardEditForm` に「Is Twinpact?」チェックボックスを追加。
        *   チェック時に呪文側編集フォーム（`SpellSideWidget`）を動的に表示するUIロジックを実装。
        *   JSONデータの `spell_side` フィールドへのネスト保存に対応。

3.  **キーワード能力とトリガー設定の整理**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `EffectEditForm` の `TriggerType` プルダウンから重複していた `S_TRIGGER` を削除。
        *   `CardEditForm` のキーワード能力チェックボックス群（`shield_trigger`, `blocker` 等）を集約し、設定フローを統一。

4.  **アクションタイプ「数字を宣言(選択)」の追加**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `ACTION_UI_CONFIG` に `DECLARE_NUMBER` (内部的には `SELECT_NUMBER`) を追加。
        *   ガチンコ・ジャッジや特定コスト指定などのために、プレイヤーが数値入力を行うUIをサポート。

5.  **未反映アクション・条件・効果のGUI反映**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `RETURN_TO_HAND` (バウンス), `SEARCH_DECK_BOTTOM`, `REVOLUTION_CHANGE` (革命チェンジ), `FRIEND_BURST`, `SELECT_NUMBER` 等のエンジン機能をエディタの選択肢に追加。
        *   `MANA_ARMED` (マナ武装), `CIVILIZATION_MATCH` 等の条件判定ロジックをエディタで設定可能にした。

6.  **カードテキスト自動生成機能**
    *   **完了日**: 2024/05/26
    *   **実装内容**:
        *   `CardTextGenerator` クラスを実装。入力されたJSONデータ（効果、対象、数値）に基づき、自然言語（日本語）のカードテキストをリアルタイムでプレビュー生成する機能を追加。

---

## Phase 1.6: エンジン機能拡張 (Completed)

エディタで作成されたデータを正しく処理するためのエンジン側対応。

1.  **ツインパクトカードのスキーマ対応**
    *   **完了日**: 2024/05/25
    *   **実装内容**:
        *   `CardDefinition` および `CardData` 構造体に `spell_side` (std::shared_ptr) フィールドを追加。
        *   `JsonLoader` による再帰的な読み込みと、Pythonバインディングを通じたアクセスを実装。

2.  **高度なメカニクス実装**
    *   **完了日**: 2024/05/30
    *   **実装内容**:
        *   **Hyper Energy**: `HyperEnergyHandler` およびコスト軽減ロジックの実装。
        *   **Revolution Change**: `RevolutionChangeHandler` および攻撃時の宣言ウィンドウ、入れ替えロジックの実装。
        *   **Just Diver**: `CardKeywords` へのフラグ追加と、選ばれない効果（`TargetUtils`）の期間制御の実装。

3.  **アクション/効果ハンドラの拡充**
    *   **完了日**: 2024/05/30
    *   **実装内容**:
        *   `ShieldBurn` (シールド焼却), `GrantKeyword` (能力付与), `SearchDeckBottom`, `ReturnToHand` 等の専用ハンドラを `src/engine/systems/card/handlers` に実装し、`GenericCardSystem` から委譲する構成を確立。
