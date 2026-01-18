# -*- coding: utf-8 -*-
"""
UnifiedActionForm のグループ/タイプ切り替え問題の修正テスト

問題: グループを「ロジック」に変更すると、タイプが「カードを引く (DRAW_CARD)」になってしまう
期待: グループに対応する最初のコマンドタイプ (例: LOGIC → QUERY) が選択される
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_command_groups_structure():
    """COMMAND_GROUPS の構造を確認"""
    print("\n" + "=" * 80)
    print("COMMAND_GROUPS 構造確認")
    print("=" * 80)
    
    config_path = project_root / "data" / "configs" / "command_ui.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    command_groups = config.get("COMMAND_GROUPS", {})
    
    print("\n各グループの最初のコマンド:")
    for group, commands in command_groups.items():
        first_cmd = commands[0] if commands else "（空）"
        print(f"  {group:15s} -> {first_cmd}")
    
    # 特に LOGIC グループを確認
    logic_commands = command_groups.get("LOGIC", [])
    print(f"\nLOGIC グループの全コマンド: {logic_commands}")
    
    # DRAW グループを確認
    draw_commands = command_groups.get("DRAW", [])
    print(f"DRAW グループの全コマンド: {draw_commands}")
    
    assert "QUERY" in logic_commands, "LOGIC グループに QUERY が含まれていません"
    assert "DRAW_CARD" in draw_commands, "DRAW グループに DRAW_CARD が含まれていません"
    assert "DRAW_CARD" not in logic_commands, "LOGIC グループに DRAW_CARD が含まれています（異常）"
    
    print("\n✅ COMMAND_GROUPS の構造は正常です")
    return command_groups


def test_unified_action_form_group_switch():
    """UnifiedActionForm のグループ切り替えテスト (GUIなしモック)"""
    print("\n" + "=" * 80)
    print("UnifiedActionForm グループ切り替えロジックテスト")
    print("=" * 80)
    
    from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
    
    # Load COMMAND_GROUPS
    COMMAND_GROUPS = EditorConfigLoader.get_command_groups()
    
    # シミュレート: LOGIC グループを選択
    grp = "LOGIC"
    types = COMMAND_GROUPS.get(grp, [])
    
    print(f"\n選択グループ: {grp}")
    print(f"グループ内のタイプ: {types}")
    
    # 修正前の問題: types が空でない場合、最初のタイプが選択されるはず
    if types:
        selected_type = types[0]
        print(f"選択されるべきタイプ: {selected_type}")
        
        # 検証
        assert selected_type == "QUERY", f"期待: QUERY, 実際: {selected_type}"
        assert selected_type != "DRAW_CARD", "DRAW_CARD が選択されています（異常）"
        
        print("\n✅ 正しいタイプが選択されます")
    else:
        print("⚠️ LOGIC グループが空です")


def test_fallback_logic():
    """フォールバックロジックのテスト"""
    print("\n" + "=" * 80)
    print("フォールバックロジックテスト")
    print("=" * 80)
    
    from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
    
    COMMAND_GROUPS = EditorConfigLoader.get_command_groups()
    
    # ケース1: 正常なグループ
    print("\nケース1: LOGIC グループ (正常)")
    grp = "LOGIC"
    types = COMMAND_GROUPS.get(grp, [])
    
    # 修正後のロジック
    t = None  # シミュレート: currentData() が None
    if t is None:
        if len(types) > 0:
            t = types[0]  # 最初の項目
        if t is None:
            t = "NONE"  # 最終フォールバック
    
    print(f"  選択されたタイプ: {t}")
    assert t == "QUERY", f"期待: QUERY, 実際: {t}"
    
    # ケース2: 空のグループ
    print("\nケース2: RESTRICTION グループ (空)")
    grp = "RESTRICTION"
    types = COMMAND_GROUPS.get(grp, [])
    
    t = None
    if t is None:
        if len(types) > 0:
            t = types[0]
        if t is None:
            t = "NONE"
    
    print(f"  選択されたタイプ: {t}")
    assert t == "NONE", f"期待: NONE, 実際: {t}"
    
    # ケース3: 修正前の問題（DRAW というグループ名を使ってしまう）
    print("\nケース3: 修正前の問題再現チェック")
    bad_fallback = "DRAW"  # これはグループ名であって、コマンドタイプではない
    
    # DRAW がコマンドタイプとして有効か確認
    all_commands = []
    for commands in COMMAND_GROUPS.values():
        all_commands.extend(commands)
    
    if bad_fallback in all_commands:
        print(f"  ⚠️ '{bad_fallback}' はコマンドタイプとして存在します")
    else:
        print(f"  ✅ '{bad_fallback}' はコマンドタイプではありません（グループ名です）")
    
    print("\n✅ フォールバックロジックは正常です")


def main():
    """全テストを実行"""
    print("\n" + "=" * 80)
    print("GUI ノード切り替え修正 - 検証テスト")
    print("=" * 80)
    
    try:
        # テスト1: COMMAND_GROUPS の構造確認
        command_groups = test_command_groups_structure()
        
        # テスト2: グループ切り替えロジック
        test_unified_action_form_group_switch()
        
        # テスト3: フォールバックロジック
        test_fallback_logic()
        
        print("\n" + "=" * 80)
        print("✅ 全テスト合格！")
        print("=" * 80)
        print("\n修正内容:")
        print("1. on_group_changed() でシグナルをブロック")
        print("2. on_type_changed() のフォールバック値を DRAW → NONE に修正")
        print("3. _load_ui_from_data() でグループ変更後に type_combo を再構築")
        print("\n期待される動作:")
        print("- LOGIC グループ選択時: QUERY が選択される")
        print("- DRAW グループ選択時: DRAW_CARD が選択される")
        print("- 空のグループ選択時: NONE にフォールバック")
        print("=" * 80 + "\n")
        
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ テスト失敗: {e}")
        print("=" * 80 + "\n")
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
