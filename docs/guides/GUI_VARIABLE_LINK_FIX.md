````markdown
# GUI変数リンク機能の修正完了

## 修正内容

カード移動コマンドで出力された変数を入力ソースとして参照できるよう、以下を修正しました：

### 1. UnifiedActionFormの修正 ([unified_action_form.py](dm_toolkit/gui/editor/forms/unified_action_form.py))

#### 変更点:
- **'links'キーの処理を追加**: スキーマで使用される'links'フィールドを'input_link'/'output_link'と同様に処理
- **current_itemの設定**: VariableLinkWidgetが前のステップの出力変数を取得できるよう、current_itemを設定
- **produces_outputヒントの設定**: VariableLinkWidgetにproduces_outputフラグを渡し、Output Keyフィールドの表示を制御

#### 具体的な修正:

```python
# _create_widget_for_field: produces_outputヒントを設定
if field_schema.field_type == FieldType.LINK and hasattr(widget, 'set_output_hint'):
    widget.set_output_hint(field_schema.produces_output)

# _load_ui_from_data: current_itemを設定
for key, widget in self.widgets_map.items():
    if key == 'links' or key == 'input_link' or key == 'output_link':
        if hasattr(widget, 'set_current_item'):
            widget.set_current_item(item)

# データ処理で'links'キーを認識
if key == 'links' or key == 'input_link' or key == 'output_link':
    widget.set_value(data)  # または widget.get_value()
```

### 2. 出力される変数の構造

各カード移動コマンド実行後、以下の変数が利用可能になります：

| 変数名 | 内容 | 例 |
|--------|------|-----|
| `{output_value_key}` | 移動枚数 | `destroyed = 3` |
| `{output_value_key}_ids_0` | 1枚目のカードID | `destroyed_ids_0 = 1234` |
| `{output_value_key}_ids_1` | 2枚目のカードID | `destroyed_ids_1 = 5678` |
| `{output_value_key}_ids_count` | ID配列の長さ | `destroyed_ids_count = 3` |

### 3. GUIでの使用方法

#### ステップ1: カード移動コマンドを追加
1. カードエディタを開く: `python python/gui/card_editor.py`
2. 効果に新しいコマンドを追加（例: DESTROY）
3. **Variable Links**セクションが表示されます

#### ステップ2: 出力キーの確認
- **Output Key**フィールドに自動生成されたキー（例: `var_DESTROY_0`）が表示されます
- このキーが次のステップで参照可能な変数名になります

#### ステップ3: 出力変数を参照
1. 次のコマンド（例: DRAW_CARD）を追加
2. **Variable Links**セクションの**Input Source**ドロップダウンをクリック
3. 前のステップの出力が選択肢に表示されます：
   - `Step 0: DESTROY -> destroyed`（例）
4. 使用方法を**Input Usage**で選択：
   - `Amount`: 破壊した枚数分ドロー
   - `Cost`: 破壊した枚数をコスト計算に使用
   - など

### 4. 動作確認

以下のコマンドでGUIを起動し、実際に動作を確認してください：

```powershell
.\.venv\Scripts\python.exe python/gui/card_editor.py
```

#### 確認項目:
- ✓ DESTROY等のコマンド追加時に**Variable Links**フィールドが表示される
- ✓ **Output Key**が自動生成される（`var_DESTROY_0`など）
- ✓ 次のコマンドで**Input Source**に前のステップの出力が選択肢として表示される
- ✓ **Input Usage**で使用方法を選択できる

### 5. トラブルシューティング

#### Variable Linksフィールドが表示されない
- スキーマが正しく登録されているか確認: `python verify_card_movement_output.py`
- コマンドタイプを再選択してみる（DESTROY → 別のコマンド → DESTROY）

#### 出力変数が選択肢に表示されない
- 前のステップに`output_value_key`が設定されているか確認
- GUIを再起動して、カードJSONを再ロード

#### Output Keyが自動生成されない
- スキーマで`produces_output=True`が設定されているか確認
- `f_links_out`が使用されているか確認（`f_links_in`では出力されません）

### 6. カードID変数の将来対応

現在、`{key}_ids_0`, `{key}_ids_1`などのカードID変数は`execution_context`に格納されていますが、GUIでの直接参照はまだサポートされていません。

将来的に以下の機能を追加予定：
- カードID配列の参照（`destroyed_ids[0]`など）
- カード属性の条件判定（「コスト5以上のカードを破壊した場合」など）

## 検証

以下のスクリプトで修正が正しく適用されているか確認できます：

```powershell
# スキーマ確認
.\.venv\Scripts\python.exe verify_card_movement_output.py

# 出力フォーマット確認
.\.venv\Scripts\python.exe verify_output_format.py
```

## 関連ファイル

- [schema_config.py](dm_toolkit/gui/editor/schema_config.py): f_links_out設定
- [unified_action_form.py](dm_toolkit/gui/editor/forms/unified_action_form.py): linksキー処理
- [variable_link_widget.py](dm_toolkit/gui/editor/forms/parts/variable_link_widget.py): ウィジェット実装
- [command_system.cpp](src/engine/systems/command_system.cpp): C++出力実装
- [CARD_MOVEMENT_OUTPUT.md](docs/CARD_MOVEMENT_OUTPUT.md): 機能詳細ドキュメント

````
