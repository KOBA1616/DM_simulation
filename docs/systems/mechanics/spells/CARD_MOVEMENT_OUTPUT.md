````markdown
# カード移動コマンドの出力機能

## 概要

カード移動関連のコマンド（TRANSITION, DESTROY, DISCARD, RETURN_TO_HAND, MANA_CHARGE等）は、実行結果を変数に出力できるようになりました。

## 対応コマンド

以下のコマンドで出力機能が有効化されています：

- **TRANSITION**: 汎用カード移動
- **DESTROY**: クリーチャー破壊
- **DISCARD**: 手札破棄
- **RETURN_TO_HAND**: 手札に戻す
- **MANA_CHARGE** / **ADD_MANA**: マナチャージ
- **BREAK_SHIELD**: シールドブレイク（既存）

## 出力される情報

各コマンド実行後、以下のデータが `execution_context` に格納されます：

### 1. 移動枚数（既存）
- キー: `output_value_key`
- 型: `int`
- 値: 実際に移動したカード枚数

### 2. カードインスタンスIDリスト（新規）
- キー接頭辞: `{output_value_key}_ids`
- 個別キー: `{output_value_key}_ids_0`, `{output_value_key}_ids_1`, ...
- カウント: `{output_value_key}_ids_count`
- 型: `int` (各インスタンスID)

## データ構造

```cpp
// 例: 2枚のクリーチャーを破壊した場合
execution_context["destroyed"] = 2;                     // 破壊枚数
execution_context["destroyed_ids_count"] = 2;           // IDリスト長
execution_context["destroyed_ids_0"] = 1234;            // 1枚目のインスタンスID
execution_context["destroyed_ids_1"] = 5678;            // 2枚目のインスタンスID
```

## GUI での設定

スキーマ更新により、移動系コマンドは自動的に `produces_output=True` となり、GUIで `output_value_key` が生成されます。

```python
# dm_toolkit/gui/editor/schema_config.py
register_schema(CommandSchema("DESTROY", [
    f_target,
    f_filter,
    FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
    f_links_out  # 出力有効化
]))
```

## 使用例1: 破壊したカード数に応じたドロー

```json
{
  "effects": [
    {
      "trigger": "ON_SUMMON",
      "commands": [
        {
          "type": "DESTROY",
          "target_group": "PLAYER_OPPONENT",
          "target_filter": {"types": ["CREATURE"]},
          "amount": 3,
          "output_value_key": "destroyed"
        },
        {
          "type": "DRAW_CARD",
          "target_group": "PLAYER_SELF",
          "input_value_key": "destroyed",
          "input_value_usage": "AMOUNT"
        }
      ]
    }
  ]
}
```

**動作**: 相手のクリーチャーを3体まで破壊し、破壊した数だけカードを引く。

## 使用例2: 手札に戻したカードの再利用

```json
{
  "effects": [
    {
      "trigger": "ON_ATTACK",
      "commands": [
        {
          "type": "RETURN_TO_HAND",
          "target_group": "PLAYER_SELF",
          "target_filter": {"types": ["CREATURE"]},
          "from_zone": "BATTLE",
          "amount": 1,
          "output_value_key": "returned"
        },
        {
          "type": "CONDITION",
          "condition": {
            "type": "VARIABLE_GT",
            "variable": "returned",
            "value": 0
          },
          "if_true": [
            {
              "type": "DRAW_CARD",
              "target_group": "PLAYER_SELF",
              "amount": 1
            }
          ]
        }
      ]
    }
  ]
}
```

**動作**: 自分のクリーチャー1体を手札に戻す。戻した場合、カードを1枚引く。

## 使用例3: マナチャージ後の条件確認

```json
{
  "effects": [
    {
      "trigger": "ON_SUMMON",
      "commands": [
        {
          "type": "MANA_CHARGE",
          "target_group": "PLAYER_SELF",
          "amount": 2,
          "output_value_key": "charged"
        },
        {
          "type": "CONDITION",
          "condition": {
            "type": "VARIABLE_EQ",
            "variable": "charged",
            "value": 2
          },
          "if_true": [
            {
              "type": "ADD_SHIELD",
              "target_group": "PLAYER_SELF",
              "amount": 1
            }
          ]
        }
      ]
    }
  ]
}
```

**動作**: 山札の上から2枚をマナゾーンに置く。2枚置けた場合、新しいシールドを1つ追加。

## 将来の拡張: カードID参照

現在、カードIDは `execution_context` に格納されていますが、Python側でまだ直接参照できません。

将来的に以下の機能を追加予定：

```json
{
  "type": "CONDITION",
  "condition": {
    "type": "CARD_ATTRIBUTE_CHECK",
    "variable": "destroyed_ids",
    "index": 0,
    "attribute": "cost",
    "operator": "GTE",
    "value": 5
  }
}
```

これにより、「コスト5以上のカードを破壊した場合」のような高度な条件判定が可能になります。

## 技術詳細

### C++ 実装箇所

-- [src/engine/systems/command_system.cpp](../../src/engine/systems/command_system.cpp)
  - `DESTROY` (line ~325-350)
  - `DISCARD` (line ~355-380)
  - `RETURN_TO_HAND` (line ~385-420)
  - `MANA_CHARGE` / `ADD_MANA` (line ~275-300)
  - `TRANSITION` (line ~178-210)

### Python スキーマ

-- [dm_toolkit/gui/editor/schema_config.py](../../dm_toolkit/gui/editor/schema_config.py)
  - DESTROY, MANA_CHARGE, RETURN_TO_HAND, BREAK_SHIELD (line ~40-48)
  - TRANSITION (line ~60-70)

## 注意事項

1. **execution_context の制限**: 現在は `std::map<std::string, int>` のため、整数値のみ格納可能
2. **IDリストのアクセス**: Python側で `{key}_ids_0`, `{key}_ids_1` ... を順次読み取る必要がある
3. **移動失敗時**: 移動に失敗したカードはカウントもIDリストも含まれない

## 検証方法

C++拡張をビルド後、以下で動作確認：

```powershell
# ビルド（必要な場合）
scripts/build.ps1 -Config Release

# GUIでカード編集
C:\Users\ichirou\DM_simulation\.venv\Scripts\python.exe python/gui/card_editor.py
```

GUIで移動系コマンドを追加すると、自動的に `output_value_key` が生成され、実行時にカードIDも記録されます。

````
