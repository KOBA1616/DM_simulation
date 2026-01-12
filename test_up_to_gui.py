# -*- coding: utf-8 -*-
"""Simple GUI test for up_to field visibility"""

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import sys

def main():
    app = QApplication(sys.argv)
    
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    
    register_all_schemas()
    
    window = QMainWindow()
    window.setWindowTitle("UnifiedActionForm Test - DISCARD up_to field")
    
    # Create central widget
    central = QWidget()
    layout = QVBoxLayout(central)
    
    # Create form
    form = UnifiedActionForm()
    layout.addWidget(form)
    
    window.setCentralWidget(central)
    window.resize(600, 800)
    
    # Set to CARD_MOVE group and DISCARD command
    for i in range(form.action_group_combo.count()):
        if form.action_group_combo.itemData(i) == "CARD_MOVE":
            form.action_group_combo.setCurrentIndex(i)
            break
    
    for i in range(form.type_combo.count()):
        if form.type_combo.itemData(i) == "DISCARD":
            form.type_combo.setCurrentIndex(i)
            break
    
    print("✓ Window setup complete")
    print(f"✓ Widgets in map: {list(form.widgets_map.keys())}")
    print(f"✓ Optional widget: {form.widgets_map.get('optional')}")
    print(f"✓ Up To widget: {form.widgets_map.get('up_to')}")
    
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
