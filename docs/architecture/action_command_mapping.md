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

... (略、元ファイルを移動しました)

```
