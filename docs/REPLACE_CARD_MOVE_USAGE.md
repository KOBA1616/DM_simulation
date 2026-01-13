# REPLACE_CARD_MOVE 機能説明

## 概要

`REPLACE_CARD_MOVE` は、カードが本来行くはずのゾーンを別のゾーンに変更するコマンドです。
例：「墓地に置くかわりに山札の下に置く」のような効果を実装できます。

## GUI での設定方法

### カードエディタでの使用

1. **アクショングループ**: `カード移動 (CARD_MOVE)` を選択
2. **アクションタイプ**: `置換カード移動 (REPLACE_CARD_MOVE)` を選択
3. **設定項目**:
   - **From Zone (元のゾーン)**: カードが本来行くはずのゾーン (例: `GRAVEYARD`)
   - **To Zone (置換先ゾーン)**: 実際に移動するゾーン (例: `DECK_BOTTOM`)
   - **Input Link**: 前のコマンドからカードインスタンスIDを受け取る場合に設定
   - **Target Filter**: カードを選択する条件（オプション）
   - **Amount**: 移動するカードの数（デフォルト: 1）

### 設定可能なゾーン

- `GRAVEYARD` - 墓地
- `DECK` - 山札
- `DECK_TOP` - 山札の上
- `DECK_BOTTOM` - 山札の下
- `HAND` - 手札
- `BATTLE_ZONE` - バトルゾーン
- `MANA_ZONE` - マナゾーン
- `SHIELD_ZONE` - シールドゾーン

## 使用例

### 例1: 墓地に置くかわりに山札の下に置く

```json
{
  "type": "REPLACE_CARD_MOVE",
  "from_zone": "GRAVEYARD",
  "to_zone": "DECK_BOTTOM",
  "instance_id": 123,
  "amount": 1
}
```

**生成テキスト**: 「そのカードを墓地に置くかわりに山札の下に置く。」

### 例2: 入力値参照を使用（コマンドチェーン）

カード選択からの連鎖：

**コマンド1: カードを選択**
```json
{
  "type": "SEARCH_DECK",
  "target_filter": {
    "zones": ["DECK"],
    "types": ["CREATURE"]
  },
  "amount": 1,
  "output_value_key": "selected_card"
}
```

**コマンド2: 選択されたカードを置換移動**
```json
{
  "type": "REPLACE_CARD_MOVE",
  "from_zone": "GRAVEYARD",
  "to_zone": "DECK_BOTTOM",
  "input_value_key": "selected_card",
  "amount": 1
}
```

**生成テキスト**: 
1. 「山札からクリーチャーを1枚選ぶ。」
2. 「そのカードを墓地に置くかわりに山札の下に置く。」

## コマンド構造

```typescript
interface ReplaceCardMoveCommand {
  type: "REPLACE_CARD_MOVE";
  
  // 必須フィールド
  from_zone: string;           // 本来の移動先ゾーン
  to_zone: string;             // 実際の移動先ゾーン
  
  // オプションフィールド
  instance_id?: number;        // 特定のカードインスタンスID
  input_value_key?: string;    // 前のコマンドからの入力参照
  target_filter?: FilterDef;   // カード選択条件
  amount?: number;             // 移動する枚数 (デフォルト: 1)
  owner_id?: number;           // プレイヤーID
  output_value_key?: string;   // 次のコマンドへの出力キー
}
```

## 実装の詳細

### action_to_command.py での処理

`_handle_replace_card_move()` 関数が以下を処理します：

1. `from_zone` を `original_to_zone` として保存
2. `to_zone` を置換先として設定
3. `instance_id` や `input_value_key` を保持
4. テキスト生成用のメタデータを整理

### エンジン実行

- **EffectResolver**: エフェクトチェーン内でコンテキストを管理し、`input_value_key` を解決
- **EngineCompat**: フォールバックパスで `instance_id` を使用してカードを移動

### テキスト生成

`CardTextGenerator._format_command()` が以下のテンプレートを使用：

```
"{target}を{orig}に置くかわりに{dest}に置く。"
```

- `target`: `input_value_key` がある場合は "そのカード"、それ以外はフィルタから生成
- `orig`: `from_zone` のローカライズ名
- `dest`: `to_zone` のローカライズ名

## テスト

テストケースは以下で確認できます：

- `python/tests/unit/test_text_generator.py` - テキスト生成テスト
- `python/tests/unit/converter/test_action_to_command.py` - コマンド変換テスト

テスト実行：

```bash
python -m pytest python/tests/unit/test_text_generator.py -k replace_card_move -v
```

## 関連ファイル

- `dm_toolkit/action_to_command.py` - コマンド変換ロジック
- `dm_toolkit/gui/editor/configs/action_config.py` - UI設定
- `dm_toolkit/gui/editor/text_generator.py` - テキスト生成
- `dm_toolkit/engine/compat.py` - エンジン実行
- `dm_toolkit/gui/i18n.py` - ゾーン名の翻訳

## 注意事項

1. **入力値参照**: `input_value_key` を使用する場合、前のコマンドで `output_value_key` を設定する必要があります
2. **ゾーン名**: UI では長い名前（`BATTLE_ZONE`）と短い名前（`BATTLE`）の両方が使用可能です
3. **翻訳**: ゾーン名は自動的に日本語に翻訳されます（例: `DECK` → "デッキ" または "山札"）
4. **実行順序**: エフェクトチェーン内で使用する場合、コマンドの順序が重要です

## 今後の拡張

将来的には以下の機能を追加する可能性があります：

- 条件付き置換（例：「パワー3000以下なら墓地、それ以外は山札の下」）
- 複数カードの一括置換
- カスタム置換ロジック（スクリプト機能）
