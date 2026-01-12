# -*- coding: utf-8 -*-
"""Test script to verify up_to field detection in UnifiedActionForm"""

from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    from dm_toolkit.gui.editor.schema_def import get_schema
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    
    register_all_schemas()
    
    # Create form
    form = UnifiedActionForm()
    
    # Print groups
    print("\n=== Available groups ===")
    for i in range(form.action_group_combo.count()):
        print(f"  {i}: {form.action_group_combo.itemData(i)} - {form.action_group_combo.itemText(i)}")
    
    # Set command type to DISCARD
    print("\n=== Available command types ===")
    for i in range(form.type_combo.count()):
        print(f"  {i}: {form.type_combo.itemData(i)} - {form.type_combo.itemText(i)}")
    
    print("\n=== Setting command type to DISCARD ===")
    # Find DISCARD in type_combo
    found = False
    for i in range(form.type_combo.count()):
        if form.type_combo.itemData(i) == "DISCARD":
            print(f"Found DISCARD at index {i}")
            form.type_combo.setCurrentIndex(i)
            found = True
            break
    
    if not found:
        print("DISCARD not found! Trying to set group to CARD_MOVE...")
        # Try setting group first
        for i in range(form.action_group_combo.count()):
            grp_data = form.action_group_combo.itemData(i)
            print(f"  Checking group {i}: {grp_data}")
            if grp_data == "CARD_MOVE":
                form.action_group_combo.setCurrentIndex(i)
                print(f"  -> Set group to {grp_data}")
                break
        
        print("\n=== Available command types after group change ===")
        for i in range(form.type_combo.count()):
            print(f"  {i}: {form.type_combo.itemData(i)} - {form.type_combo.itemText(i)}")
        
        # Try finding DISCARD again
        for i in range(form.type_combo.count()):
            if form.type_combo.itemData(i) == "DISCARD":
                print(f"Found DISCARD at index {i}")
                form.type_combo.setCurrentIndex(i)
                found = True
                break
    
    print(f"\n=== After setting to DISCARD ===")
    print(f"Current type: {form.type_combo.currentData()}")
    
    # Check widgets_map
    print(f"\nWidgets in widgets_map: {list(form.widgets_map.keys())}")
    
    # Check if up_to widget exists
    if 'up_to' in form.widgets_map:
        print("OK: up_to widget found in widgets_map")
        print(f"  up_to widget type: {type(form.widgets_map['up_to'])}")
    else:
        print("NG: up_to widget NOT found in widgets_map")
    
    if 'optional' in form.widgets_map:
        print("OK: optional widget found in widgets_map")
    else:
        print("NG: optional widget NOT found in widgets_map")
    
    form.show()
    app.exec()

if __name__ == "__main__":
    main()
