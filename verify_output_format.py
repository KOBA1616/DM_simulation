#!/usr/bin/env python
"""
カードID/枚数出力の詳細検証

各コマンドで以下が出力されることを確認：
1. execution_context[output_value_key] = 移動枚数（int）
2. execution_context[output_value_key + "_ids_0"] = 1枚目のカードID（int）
3. execution_context[output_value_key + "_ids_1"] = 2枚目のカードID（int）
   ...
4. execution_context[output_value_key + "_ids_count"] = ID総数（int）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_cpp_implementation():
    """C++コードで両方出力されているか検証"""
    
    print("=" * 80)
    print("カードID/枚数 出力実装検証")
    print("=" * 80)
    
    cpp_file = project_root / "src" / "engine" / "systems" / "command_system.cpp"
    
    if not cpp_file.exists():
        print(f"✗ {cpp_file} が見つかりません")
        return False
    
    content = cpp_file.read_text(encoding='utf-8')
    
    # 各コマンドの実装パターンをチェック
    commands = {
        "DESTROY": ("destroyed", "destroyed_ids"),
        "DISCARD": ("discarded", "discarded_ids"),
        "RETURN_TO_HAND": ("returned", "returned_ids"),
        "BOOST_MANA": ("charged", "charged_ids"),
        "TRANSITION": ("moved_count", "moved_ids")
    }
    
    print("\n各コマンドの出力実装:")
    print("-" * 80)
    
    all_ok = True
    for cmd_name, (count_var, ids_var) in commands.items():
        # 1. カウント変数の宣言を確認
        count_declaration = f"int {count_var} = 0;" in content
        
        # 2. IDベクター変数の宣言を確認
        ids_declaration = f"std::vector<int> {ids_var};" in content
        
        # 3. execution_contextへのカウント格納を確認
        count_output = f"execution_context[cmd.output_value_key] = {count_var};" in content
        
        # 4. IDのループ格納を確認
        ids_loop = f'execution_context[ids_key + "_" + std::to_string(i)] = {ids_var}[i];' in content
        
        # 5. カウント格納を確認
        count_store = f'execution_context[ids_key + "_count"] = static_cast<int>({ids_var}.size());' in content
        
        status = "✓" if all([count_declaration, ids_declaration, count_output, ids_loop, count_store]) else "✗"
        
        print(f"\n{status} {cmd_name}:")
        print(f"  - カウント変数宣言: {'✓' if count_declaration else '✗'} int {count_var} = 0;")
        print(f"  - IDベクター宣言: {'✓' if ids_declaration else '✗'} std::vector<int> {ids_var};")
        print(f"  - カウント出力: {'✓' if count_output else '✗'} execution_context[key] = {count_var};")
        print(f"  - IDループ出力: {'✓' if ids_loop else '✗'} execution_context[key_ids_i] = {ids_var}[i];")
        print(f"  - ID総数出力: {'✓' if count_store else '✗'} execution_context[key_ids_count] = size();")
        
        if not all([count_declaration, ids_declaration, count_output, ids_loop, count_store]):
            all_ok = False
    
    print("\n" + "=" * 80)
    
    # 出力データ形式の説明
    print("\n出力データ形式:")
    print("-" * 80)
    print("各コマンド実行後、execution_contextに以下のキーが追加されます：")
    print("")
    print("  1. {output_value_key}           : int    - 実際に移動した枚数")
    print("  2. {output_value_key}_ids_0     : int    - 1枚目のカードインスタンスID")
    print("  3. {output_value_key}_ids_1     : int    - 2枚目のカードインスタンスID")
    print("     ... (移動した枚数分)")
    print("  4. {output_value_key}_ids_count : int    - ID配列の総数")
    print("")
    print("例: 3体のクリーチャーを破壊し、output_value_key=\"destroyed\"の場合")
    print("")
    print("  execution_context[\"destroyed\"]           = 3     // 破壊枚数")
    print("  execution_context[\"destroyed_ids_0\"]     = 1234  // 1体目のID")
    print("  execution_context[\"destroyed_ids_1\"]     = 5678  // 2体目のID")
    print("  execution_context[\"destroyed_ids_2\"]     = 9012  // 3体目のID")
    print("  execution_context[\"destroyed_ids_count\"] = 3     // ID総数")
    print("")
    print("=" * 80)
    
    if all_ok:
        print("\n✓ 全コマンドでID/枚数の両方が出力されています")
        return True
    else:
        print("\n✗ 一部のコマンドで出力が不完全です")
        return False


if __name__ == "__main__":
    success = verify_cpp_implementation()
    sys.exit(0 if success else 1)
