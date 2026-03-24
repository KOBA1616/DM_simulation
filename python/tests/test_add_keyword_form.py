# -*- coding: utf-8 -*-
"""
ADD_KEYWORDフォームのstr_val/duration/amount機能テスト
再発防止: 数量フィールドを「非表示」ラベルへ戻すと、GUI上は見えていても
設定ファイル由来のスキーマ検証と実表示が再び食い違うため、amount も固定で検証する。
"""
import json

def test_add_keyword_schema():
    """ADD_KEYWORDスキーマの定義を確認"""
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    from dm_toolkit.gui.editor.schema_def import get_schema
    
    # スキーマを登録
    register_all_schemas()
    
    # ADD_KEYWORDスキーマを取得
    schema = get_schema("ADD_KEYWORD")
    assert schema is not None, "ADD_KEYWORD schema not found"
    
    # フィールドリストを確認
    field_keys = [f.key for f in schema.fields]
    print(f"ADD_KEYWORD fields: {field_keys}")
    
    assert "str_val" in field_keys, "str_val field not found"
    assert "duration" in field_keys, "duration field not found"
    assert "amount" in field_keys, "amount field not found"
    assert "target_group" in field_keys, "target_group field not found"
    
    # durationフィールドの選択肢を確認
    duration_field = next(f for f in schema.fields if f.key == "duration")
    amount_field = next(f for f in schema.fields if f.key == "amount")
    print(f"Duration options: {duration_field.options}")
    assert "PERMANENT" in duration_field.options, "PERMANENT not in duration options"
    assert amount_field.default == 0, f"Expected amount default=0, got {amount_field.default}"
    assert amount_field.min_value == 0, f"Expected amount min_value=0, got {amount_field.min_value}"
    
    print("✅ ADD_KEYWORD schema test passed")


def test_duration_translation():
    """DURATION翻訳を確認"""
    from dm_toolkit.gui.editor.text_resources import CardTextResources
    
    # PERMANENT翻訳を確認
    perm_text = CardTextResources.get_duration_text("PERMANENT")
    print(f"PERMANENT translation: {perm_text}")
    assert perm_text == "常に", f"Expected '常に', got '{perm_text}'"
    
    # その他の翻訳も確認
    this_turn = CardTextResources.get_duration_text("THIS_TURN")
    print(f"THIS_TURN translation: {this_turn}")
    assert this_turn == "このターン", f"Expected 'このターン', got '{this_turn}'"
    
    print("✅ Duration translation test passed")


def test_keyword_translation():
    """キーワード翻訳を確認"""
    from dm_toolkit.gui.editor.text_resources import CardTextResources
    
    # S_TRIGGER翻訳を確認
    s_trigger = CardTextResources.get_keyword_text("S_TRIGGER")
    print(f"S_TRIGGER translation: {s_trigger}")
    assert s_trigger == "S・トリガー", f"Expected 'S・トリガー', got '{s_trigger}'"
    
    # 小文字版も確認
    s_trigger_lower = CardTextResources.get_keyword_text("s_trigger")
    print(f"s_trigger translation: {s_trigger_lower}")
    assert s_trigger_lower == "S・トリガー", f"Expected 'S・トリガー', got '{s_trigger_lower}'"
    
    print("✅ Keyword translation test passed")


def test_text_generation_with_duration():
    """duration付きADD_KEYWORDのテキスト生成を確認"""
    from dm_toolkit.gui.editor.text_generator import CardTextGenerator
    
    gen = CardTextGenerator()
    
    # ID=7カードを読み込み
    with open('data/cards.json', 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    card7 = next(c for c in cards if c.get('id') == 7)
    print(f"\nTesting card: {card7['name']}")
    
    # テキスト生成
    text = gen.generate_body_text(card7)
    print(f"\nGenerated text:\n{text}")
    
    # 期待される文字列が含まれているか確認
    assert "常に、カードに「S・トリガー」を与える" in text, "Duration and keyword not found in generated text"
    assert "墓地の呪文の数が3以上なら" in text, "COMPARE_INPUT condition not found"
    
    print("\n✅ Text generation with duration test passed")


def test_command_ui_config():
    """command_ui.jsonの設定を確認"""
    from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
    config = EditorConfigLoader.get_command_ui_config()
    add_keyword_config = config.get("ADD_KEYWORD")
    assert add_keyword_config is not None, "ADD_KEYWORD not found in command_ui.json"
    
    visible = add_keyword_config.get("visible", [])
    print(f"ADD_KEYWORD visible fields: {visible}")
    
    assert "str_val" in visible, "str_val not in visible fields"
    assert "duration" in visible, "duration not in visible fields"
    assert "amount" in visible, "amount not in visible fields"
    
    labels = add_keyword_config.get("labels", {})
    assert "str_val" in labels, "str_val label not found"
    assert labels["str_val"] == "キーワード能力", f"Expected 'キーワード能力', got '{labels['str_val']}'"
    assert labels.get("amount") == "対象数（0 = すべて）", f"Unexpected amount label: {labels.get('amount')}"
    
    print("✅ Command UI config test passed")


def test_legacy_command_schema_amount_field():
    """旧command_schema.jsonも数量設定に追従していることを確認"""
    with open('data/configs/command_schema.json', 'r', encoding='utf-8') as f:
        schema = json.load(f)

    add_keyword = schema.get('ADD_KEYWORD')
    assert add_keyword is not None, 'ADD_KEYWORD not found in command_schema.json'

    fields = add_keyword.get('fields', [])
    field_map = {field.get('key'): field for field in fields}

    # 再発防止: 旧スキーマに duration 用 amount ラベルが残ると外部検証系で数量指定が誤解される。
    assert 'amount' in field_map, 'amount field not found in command_schema.json'
    assert field_map['amount'].get('label') == 'Target Count (0 = all)'
    assert 'duration' in field_map, 'duration field not found in command_schema.json'


if __name__ == "__main__":
    print("=== Testing ADD_KEYWORD Form Configuration ===\n")
    
    try:
        test_add_keyword_schema()
        print()
        test_duration_translation()
        print()
        test_keyword_translation()
        print()
        test_command_ui_config()
        print()
        test_text_generation_with_duration()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise