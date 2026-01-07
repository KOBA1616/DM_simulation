```markdown
# Action → Command マッピング (Schema Definition)

このドキュメントは、システム内で利用される Command 辞書の正規スキーマ（Schema）と、旧 Action/EffectActionType からの変換ルールを定義します。

## 1. Command Dictionary Schema

全ての Command 辞書は以下の共通フィールドを持ちます。

| Key | Type | Description |
| :-- | :-- | :-- |
| `type` | String | Command種別（必須）。例: `TRANSITION`, `MUTATE`, `QUERY` |
| `uid` | String | ユニークID（必須）。通常は UUID 文字列。 |
| `legacy_warning` | Boolean | 変換不完全な場合の警告フラグ（`True` の場合、変換要確認）。 |
| `input_value_key` | String | (Optional) 直前の処理結果を受け取る変数キー。 |
| `output_value_key` | String | (Optional) この処理結果を格納する変数キー。 |

### Command Type Definition

以下に主要な Command Type とその固有フィールドを定義します。

... (内容は既存のドキュメントと同一)

---

作成日: 2025-12-27 (Updated)

## Editor-only ノードと未対応タイプ

- `BRANCH`（条件分岐）: エディタ内ロジック用。エンジン実行コマンドではありません。保存時は `editor_only: true` を付与し、実行系へは流さないでください。
- `FLOW: SEQUENCE`（順次実行）: エディタ構造表現用。`FlowType` に標準化された `PHASE_CHANGE` / `TURN_CHANGE` / `SET_ACTIVE_PLAYER` 等のみがネイティブサポート対象です。

## テンプレート運用ガイドライン

- 生成コマンドは必ず `type` に `TRANSITION` / `MUTATE` / `FLOW` / `QUERY` / `DECIDE` / `DECLARE_REACTION` / `STAT` / `GAME_RESULT` の正規タイプを使用してください。
- ゾーン名は `HAND` / `DECK` / `BATTLE` / `MANA` / `SHIELD` / `GRAVEYARD` / `BUFFER` / `UNDER_CARD` のいずれかに正規化してください。
- ドロー表現は `TRANSITION` の `from_zone=DECK` / `to_zone=HAND` で統一し、自然文生成で「カードをN枚引く。」に変換します（`DRAW_CARD` は内部互換として扱われます）。
- マッピングは `dm_toolkit.action_to_command.action_to_command`（`map_action`）を唯一の入口として利用し、互換ロジックは `dm_toolkit.compat_wrappers` と `dm_toolkit.unified_execution` に集約します。

```
