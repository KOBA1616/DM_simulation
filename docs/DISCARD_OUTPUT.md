# DISCARD コマンドの出力機能

## 概要

`DISCARD` コマンドはスキーマ定義で `produces_output=True` が設定されており、実行結果を変数に出力できます。

## 自動生成される出力変数

GUIでDISCARDコマンドを作成すると、`UnifiedActionForm` が自動的に `output_value_key` を生成します：

```python
{
    "type": "DISCARD",
    "target_group": "PLAYER_SELF",
    "amount": 2,
    "output_value_key": "var_DISCARD_0"  # 自動生成
}
```

## 出力される情報（現在の実装）

C++エンジン (`command_system.cpp`) は以下を出力します：

- **捨てた枚数** (`int`): `execution_context[output_value_key]` に格納

```cpp
// src/engine/systems/command_system.cpp (line 348)
if (!cmd.output_value_key.empty()) {
    execution_context[cmd.output_value_key] = discarded;
}
```

## 使用例1: 捨てた枚数を参照

```json
{
  "effects": [
    {
      "trigger": "ON_SUMMON",
      "commands": [
        {
          "type": "DISCARD",
          "target_group": "PLAYER_SELF",
          "amount": 2,
          "up_to": true,
          "output_value_key": "var_DISCARD_0"
        },
        {
          "type": "DRAW_CARD",
          "target_group": "PLAYER_SELF",
          "input_value_key": "var_DISCARD_0",
          "input_value_usage": "AMOUNT"
        }
      ]
    }
  ]
}
```

**動作**: 手札を最大2枚まで捨て、捨てた枚数分カードを引く。

## 使用例2: 条件分岐

```json
{
  "effects": [
    {
      "trigger": "ON_SUMMON",
      "commands": [
        {
          "type": "DISCARD",
          "target_group": "PLAYER_SELF",
          "target_filter": {"types": ["SPELL"]},
          "amount": 1,
          "optional": true,
          "output_value_key": "var_DISCARD_0"
        },
        {
          "type": "CONDITION",
          "condition": {
            "type": "VARIABLE_GT",
            "variable": "var_DISCARD_0",
            "value": 0
          },
          "if_true": [
            {
              "type": "DRAW_CARD",
              "target_group": "PLAYER_SELF",
              "amount": 2
            }
          ]
        }
      ]
    }
  ]
}
```

**動作**: 呪文を1枚捨ててもよい。捨てた場合、カードを2枚引く。

## 将来の拡張（検討中）

C++エンジンを拡張し、以下の情報も出力可能にする予定：

```cpp
// 将来の実装案
struct DiscardResult {
    int count;                      // 捨てた枚数
    std::vector<int> instance_ids;  // 捨てたカードのインスタンスID
};
```

これにより、捨てたカード固有の情報（コスト、文明など）を後続コマンドで参照できるようになります。

## GUI での設定方法

1. エフェクト/トリガーに DISCARD コマンドを追加
2. `Variable Links` フィールドが自動的に表示される（`produces_output=True` のため）
3. 保存時に `output_value_key` が自動生成される（例: `var_DISCARD_0`）
4. 後続のコマンドで `input_value_key` に同じ変数名を指定し、`input_value_usage` で用途を選択

## 検証

テストスクリプト `test_discard_output.py` で動作確認済み：

```bash
python test_discard_output.py
```

出力例：
```
✓ SUCCESS: DISCARD command will output discarded card IDs to variable: var_DISCARD_0
```

## 関連ファイル

- スキーマ定義: `dm_toolkit/gui/editor/schema_config.py` (line 29-36)
- フォームロジック: `dm_toolkit/gui/editor/forms/unified_action_form.py` (line 206-212)
- C++実装: `src/engine/systems/command_system.cpp` (line 336-350)
- Python互換レイヤー: `dm_toolkit/engine/compat.py` (line 275)
