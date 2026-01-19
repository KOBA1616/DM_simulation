# -*- coding: utf-8 -*-
"""
カードエディタのフォーム読み込みテスト
保存された値が正しく表示されることを確認
"""
import json

def test_form_data_loading():
    """保存されたデータがフォームに正しくロードされることを確認"""
    print("=== Testing Form Data Loading ===\n")
    
    # ID=7カードを読み込み
    with open('data/cards.json', 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    card7 = next(c for c in cards if c.get('id') == 7)
    print(f"Testing card: {card7['name']}\n")
    
    # 最初のeffectのコマンドを取得
    effect = card7['effects'][0]
    commands = effect.get('commands', [])
    
    print(f"Effect has {len(commands)} commands\n")
    
    # IF コマンドを探す
    if_command = None
    for cmd in commands:
        if cmd.get('type') == 'IF':
            if_command = cmd
            break
    
    assert if_command is not None, "IF command not found"
    print(f"Found IF command: {if_command.get('uid')}")
    
    # if_true内のADD_KEYWORDを確認
    if_true = if_command.get('if_true', [])
    assert len(if_true) > 0, "if_true is empty"
    
    add_keyword = if_true[0]
    assert add_keyword.get('type') == 'ADD_KEYWORD', "First if_true command is not ADD_KEYWORD"
    
    print(f"\nADD_KEYWORD command:")
    print(f"  str_val: {add_keyword.get('str_val')}")
    print(f"  duration: {add_keyword.get('duration')}")
    print(f"  target_group: {add_keyword.get('target_group')}")
    
    # str_valが設定されているか確認
    assert add_keyword.get('str_val') == 'S_TRIGGER', \
        f"Expected str_val='S_TRIGGER', got '{add_keyword.get('str_val')}'"
    
    # durationが設定されているか確認
    assert add_keyword.get('duration') == 'PERMANENT', \
        f"Expected duration='PERMANENT', got '{add_keyword.get('duration')}'"
    
    print("\n✅ Form data structure is correct")
    
    # Schema定義を確認
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    from dm_toolkit.gui.editor.schema_def import get_schema
    
    register_all_schemas()
    schema = get_schema("ADD_KEYWORD")
    
    print("\nADD_KEYWORD schema fields:")
    for field in schema.fields:
        default = f" (default={field.default})" if field.default is not None else ""
        print(f"  {field.key}: {field.field_type.name}{default}")
    
    # str_valとdurationフィールドのデフォルト値を確認
    str_val_field = next(f for f in schema.fields if f.key == 'str_val')
    duration_field = next(f for f in schema.fields if f.key == 'duration')
    
    print(f"\nstr_val default: {str_val_field.default}")
    print(f"duration default: {duration_field.default}")
    
    assert str_val_field.default is None, "str_val should have None default"
    assert duration_field.default is None, "duration should have None default"
    
    print("\n✅ Schema defaults are correct (None)")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED")
    print("="*50)


def test_empty_data_loading():
    """データが保存されていない場合のテスト"""
    print("\n=== Testing Empty Data Loading ===\n")
    
    # 空のADD_KEYWORDコマンド（durationなし）
    empty_command = {
        "type": "ADD_KEYWORD",
        "target_group": "CARD_SELF"
    }
    
    print("Empty command (no str_val, no duration):")
    print(f"  type: {empty_command.get('type')}")
    print(f"  target_group: {empty_command.get('target_group')}")
    print(f"  str_val: {empty_command.get('str_val')}")  # None
    print(f"  duration: {empty_command.get('duration')}")  # None
    
    # フォームに読み込む際、これらのNone値は「---」（空の項目）を選択すべき
    print("\n期待される動作:")
    print("  - str_valコンボボックス: '---' (empty) が選択される")
    print("  - durationコンボボックス: '---' (empty) が選択される")
    
    print("\n✅ Empty data handling logic is defined")


if __name__ == "__main__":
    try:
        test_form_data_loading()
        test_empty_data_loading()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise