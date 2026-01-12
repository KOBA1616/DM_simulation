#!/usr/bin/env python
"""
GUI変数リンク表示確認スクリプト

UnifiedActionFormでf_links_out/f_links_inを持つコマンドが
VariableLinkWidgetを正しく表示するか確認します。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_variable_link_display():
    """GUI表示確認（実際のウィジェット生成をテスト）"""
    
    print("=" * 80)
    print("GUI変数リンク表示確認")
    print("=" * 80)
    
    # 1. スキーマ確認
    print("\n[1] スキーマでLINKフィールドを持つコマンド:")
    print("-" * 80)
    
    from dm_toolkit.gui.editor import schema_def
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    
    register_all_schemas()
    
    movement_commands = ["DESTROY", "DISCARD", "RETURN_TO_HAND", "MANA_CHARGE", "TRANSITION", "DRAW_CARD"]
    
    for cmd_name in movement_commands:
        schema = schema_def.get_schema(cmd_name)
        if schema:
            link_fields = [f for f in schema.fields if f.field_type == schema_def.FieldType.LINK]
            if link_fields:
                produces = link_fields[0].produces_output if link_fields else False
                print(f"  ✓ {cmd_name}: {len(link_fields)} LINK field(s), produces_output={produces}")
                for lf in link_fields:
                    print(f"      - key='{lf.key}', label='{lf.label}', produces_output={lf.produces_output}")
            else:
                print(f"  ✗ {cmd_name}: LINKフィールドなし")
        else:
            print(f"  ✗ {cmd_name}: スキーマなし")
    
    # 2. ウィジェット生成テスト
    print("\n[2] VariableLinkWidget生成テスト:")
    print("-" * 80)
    
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        from dm_toolkit.gui.editor.widget_factory import WidgetFactory
        from dm_toolkit.gui.editor.schema_def import FieldSchema, FieldType
        
        # f_links_outのテスト
        field_out = FieldSchema("links", "Variable Links", FieldType.LINK, produces_output=True)
        widget_out = WidgetFactory.create_widget(None, field_out, lambda: None)
        
        if widget_out:
            print(f"  ✓ f_links_out用ウィジェット生成成功: {widget_out.__class__.__name__}")
            # output_key_editが存在するか確認
            if hasattr(widget_out, 'output_key_edit'):
                print(f"      - output_key_edit: 存在")
            else:
                print(f"      ✗ output_key_edit: 存在しない")
            # input_key_comboが存在するか確認
            if hasattr(widget_out, 'input_key_combo'):
                print(f"      - input_key_combo: 存在")
            else:
                print(f"      ✗ input_key_combo: 存在しない")
        else:
            print(f"  ✗ f_links_out用ウィジェット生成失敗")
        
        # f_links_inのテスト
        field_in = FieldSchema("links", "Variable Links", FieldType.LINK, produces_output=False)
        widget_in = WidgetFactory.create_widget(None, field_in, lambda: None)
        
        if widget_in:
            print(f"  ✓ f_links_in用ウィジェット生成成功: {widget_in.__class__.__name__}")
        else:
            print(f"  ✗ f_links_in用ウィジェット生成失敗")
            
    except Exception as e:
        print(f"  ✗ ウィジェット生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 変数収集機能確認
    print("\n[3] 変数収集機能:")
    print("-" * 80)
    
    try:
        from dm_toolkit.gui.editor.variable_link_manager import VariableLinkManager
        
        # 変数なしで呼び出し（基本機能確認）
        vars_list = VariableLinkManager.get_available_variables(None)
        print(f"  ✓ get_available_variables呼び出し成功")
        print(f"      利用可能な変数数: {len(vars_list)}")
        for label, key in vars_list[:5]:  # 最初の5つのみ表示
            display_key = f"'{key}'" if key else "<empty>"
            print(f"      - {label}: {display_key}")
        
    except Exception as e:
        print(f"  ✗ 変数収集エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("確認完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. 実際のGUIを起動: python -m dm_toolkit.gui.editor.main")
    print("2. カード編集画面でDESTROYコマンドを追加")
    print("3. 'Variable Links'フィールドが表示されることを確認")
    print("4. Input Source, Input Usage, Output Keyの3つが編集可能か確認")
    
    return True


if __name__ == "__main__":
    success = test_variable_link_display()
    sys.exit(0 if success else 1)
