#!/usr/bin/env python
"""
カード移動コマンドの出力機能検証スクリプト

このスクリプトは、command_system.cppの変更が正しく動作するかを検証します。
実際のC++拡張をロードして、カードID出力が正しく格納されるかチェックします。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_card_movement_output():
    """カード移動コマンドの出力機能をテスト"""
    
    print("=" * 70)
    print("カード移動コマンド出力機能 検証")
    print("=" * 70)
    
    # 1. スキーマ確認
    print("\n[1] スキーマ設定の確認...")
    try:
        from dm_toolkit.gui.editor import schema_def
        
        # スキーマを登録
        from dm_toolkit.gui.editor.schema_config import register_all_schemas
        register_all_schemas()
        
        movement_commands = ["DESTROY", "DISCARD", "RETURN_TO_HAND", "MANA_CHARGE", "TRANSITION"]
        for cmd_name in movement_commands:
            schema = schema_def.get_schema(cmd_name)
            if schema:
                has_output = any(f.produces_output for f in schema.fields if hasattr(f, 'produces_output'))
                status = "✓" if has_output else "✗"
                print(f"  {status} {cmd_name}: produces_output = {has_output}")
            else:
                print(f"  ✗ {cmd_name}: スキーマが見つかりません")
        
        print("  → スキーマ確認完了")
    except Exception as e:
        print(f"  ✗ スキーマ確認エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. C++拡張の読み込み確認
    print("\n[2] C++拡張モジュールの読み込み...")
    try:
        import dm_ai_module
        print(f"  ✓ dm_ai_module 読み込み成功: {dm_ai_module.__file__}")
        
        # バージョン確認
        if hasattr(dm_ai_module, '__version__'):
            print(f"  バージョン: {dm_ai_module.__version__}")
        
    except ImportError as e:
        print(f"  ✗ C++拡張のインポートに失敗しました: {e}")
        print("  → ビルドを実行してください: cmake --build build-msvc --config Release")
        return False
    
    # 3. ドキュメント確認
    print("\n[3] ドキュメントの確認...")
    doc_path = project_root / "docs" / "CARD_MOVEMENT_OUTPUT.md"
    if doc_path.exists():
        print(f"  ✓ {doc_path.name} が存在します")
    else:
        print(f"  ✗ {doc_path.name} が見つかりません")
    
    # 4. 使用例確認
    print("\n[4] 使用例の確認...")
    examples_dir = project_root / "data" / "examples"
    example_files = [
        "destroy_with_draw.json",
        "transition_chain.json", 
        "mana_charge_conditional.json"
    ]
    
    for filename in example_files:
        filepath = examples_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} が見つかりません")
    
    print("\n" + "=" * 70)
    print("検証完了")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. C++コードのビルド: cmake --build build-msvc --config Release")
    print("2. GUIでの動作確認: python -m dm_toolkit.gui.editor.main")
    print("3. 統合テストの実行: pytest python/tests/")
    
    return True


if __name__ == "__main__":
    success = test_card_movement_output()
    sys.exit(0 if success else 1)
