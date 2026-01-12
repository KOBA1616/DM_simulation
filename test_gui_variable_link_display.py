#!/usr/bin/env python
"""
GUI変数リンク機能の動作確認

実際にUnifiedActionFormを起動して、Variable Linksフィールドが
正しく表示されることを確認します。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    
    register_all_schemas()
    
    app = QApplication(sys.argv)
    
    # メインウィンドウを作成
    window = QMainWindow()
    window.setWindowTitle("UnifiedActionForm テスト - Variable Links表示確認")
    window.resize(600, 800)
    
    # UnifiedActionFormを作成
    form = UnifiedActionForm()
    
    # テストデータ（DESTROYコマンド）
    test_data = {
        "type": "DESTROY",
        "target_group": "PLAYER_OPPONENT",
        "target_filter": {
            "types": ["CREATURE"]
        },
        "amount": 2,
        "input_value_key": "",
        "output_value_key": "var_DESTROY_0"
    }
    
    # ダミーのQTreeWidgetItemを作成（current_item用）
    from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
    from PyQt6.QtCore import Qt
    tree = QTreeWidget()
    parent_item = QTreeWidgetItem(tree, ["Effect"])
    test_item = QTreeWidgetItem(parent_item, ["Command 0"])
    test_item.setData(0, Qt.ItemDataRole.UserRole + 2, test_data)
    
    # フォームにデータをロード（itemのみを渡す）
    form.load_data(test_item)
    
    # 中央ウィジェットとして設定
    window.setCentralWidget(form)
    
    # 確認手順を表示
    print("=" * 80)
    print("GUI変数リンク機能 動作確認")
    print("=" * 80)
    print("\n確認手順:")
    print("1. ウィンドウが表示されます")
    print("2. 'Variable Links'というラベルのフィールドが表示されていることを確認")
    print("3. 以下の3つのコントロールが存在することを確認:")
    print("   - Input Source (ComboBox)")
    print("   - Input Usage (ComboBox) - Input Sourceを選択時に表示")
    print("   - Output Key (LineEdit) - 'var_DESTROY_0'が表示されているはず")
    print("\n4. Input Sourceドロップダウンをクリックして、選択肢を確認:")
    print("   - '手動入力'")
    print("   - 各種予約変数 ($evo_target, $cost_reducer等)")
    print("\n5. Output Keyフィールドに'var_DESTROY_0'が表示されていることを確認")
    print("\n6. すべて確認できたらウィンドウを閉じてください")
    print("=" * 80)
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
