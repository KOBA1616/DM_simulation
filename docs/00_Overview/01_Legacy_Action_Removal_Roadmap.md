# Legacy Action Removal Roadmap (要件定義書 01)

## 概要
本ドキュメントは、古い `Action` ベースの実装（カードJSONの `actions` 配列、`ActionType` enumの大部分）を段階的に廃止し、`GameCommand`（コマンドパターン）への完全移行を行うためのロードマップです。

## ステータス
*   [Status: WIP] Phase 1: コマンド基盤の完成と変換レイヤーの整備
*   [Status: Todo] Phase 2: ActionGenerator / EffectResolver の移行
*   [Status: Todo] Phase 3: カードJSONデータの移行ツール作成と実行
*   [Status: Todo] Phase 4: UI / Editor の Command 対応
*   [Status: Todo] Phase 5: 旧コードの削除

## 詳細要件

### Phase 1: コマンド基盤の完成と変換レイヤーの整備
1.  **GameCommandの拡充**:
    *   現在不足しているActionに対応するCommandクラスを作成する。
    *   不足リスト: `SearchDeck` (完了), `ShuffleDeck` (完了), `ShieldTrigger`, `SelectTarget` (汎用), `ResolveBattle` (詳細化) 等。
    *   `src/engine/game_command/commands.hpp` および `src/engine/game_command/commands.cpp` を更新。
2.  **Conversion Layer**:
    *   `dm_toolkit.action_to_command` モジュール（Python）または C++ 側の変換ロジックを強化し、既存の全ての `Action` を `GameCommand` 配列に変換できるようにする。

### Phase 2: ActionGenerator / EffectResolver の移行
1.  **EffectResolver**:
    *   `EffectResolver::resolve_effect` が `Action` ではなく `GameCommand` を実行するように変更する。
    *   現状の `resolve_action` は `CommandExecutor` への委譲メソッドに変更する。
2.  **ActionGenerator**:
    *   AIの行動生成 (`ActionGenerator`) は当面 `Action` を生成してもよいが、実行時には即座に Command に変換されるフローを確立する。

### Phase 3: カードJSONデータの移行ツール作成と実行
1.  **Migration Script**:
    *   `data/cards.json` を読み込み、各カードの `actions` フィールドを `commands` フィールド（新しいJSONスキーマに基づく）に変換するPythonスクリプトを作成する。
    *   `actions` 配列は廃止し、`on_play`, `trigger_abilities`, `static_abilities` 等の構造化されたフィールドに移行する（または `commands` リストとして統一）。
2.  **Data Verification**:
    *   変換後のデータで既存のテストケースが通過することを確認する。

### Phase 4: UI / Editor の Command 対応
1.  **Card Editor**:
    *   「Action」タブを「Effect/Command」タブにリニューアルし、Commandベースの編集画面にする。
    *   ただし、これはUIの大幅改修になるため、Phase 1-3完了後に実施。今回はスコープ外とする。

### Phase 5: 旧コードの削除
1.  **Cleanup**:
    *   `src/engine/action.hpp` から不要な `ActionType` を削除。
    *   `GenericCardSystem` 内の古いAction処理ロジックを削除。

## 今回の作業スコープ (Phase 1 & 2 Focus)
ユーザー指示「要件定義書01に従い、旧アクションからコマンド方式への移行を進めて...」に基づき、以下を実行する。

1.  不足している `GameCommand` の実装 (`src/engine/game_command/commands.hpp`).
    *   `SearchDeckCommand` および `ShuffleCommand` の実装とPythonバインディングを完了。
    *   ユニットテストによる動作確認済み。
2.  `EffectResolver` での `GameCommand` 実行パスの確立。
3.  検証テストの作成。

完了後、本ドキュメントのステータスを更新する。
