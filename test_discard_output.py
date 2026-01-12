# -*- coding: utf-8 -*-
"""Test script to verify DISCARD command generates output_value_key for discarded cards."""

from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    from dm_toolkit.gui.editor.schema_config import register_all_schemas
    from dm_toolkit.gui.editor.schema_def import get_schema
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    
    register_all_schemas()
    
    # Verify DISCARD schema has produces_output field
    schema = get_schema('DISCARD')
    print("\n=== DISCARD Schema Analysis ===")
    print(f"Command: DISCARD")
    print(f"Fields:")
    for field in schema.fields:
        output_flag = "✓ produces_output" if field.produces_output else ""
        print(f"  - {field.key}: {field.label} ({field.field_type}) {output_flag}")
    
    has_output = any(f.produces_output for f in schema.fields)
    print(f"\nSchema produces output: {has_output}")
    
    # Create form and populate with DISCARD
    form = UnifiedActionForm()
    
    # Find DISCARD in type combo
    for i in range(form.action_group_combo.count()):
        if form.action_group_combo.itemData(i) == "CARD_MOVE":
            form.action_group_combo.setCurrentIndex(i)
            break
    
    for i in range(form.type_combo.count()):
        if form.type_combo.itemData(i) == "DISCARD":
            form.type_combo.setCurrentIndex(i)
            break
    
    # Simulate save to trigger output_value_key generation
    test_data = {}
    form._save_ui_to_data(test_data)
    
    print("\n=== Generated Command Data ===")
    print(f"Type: {test_data.get('type')}")
    print(f"Output Value Key: {test_data.get('output_value_key', '❌ NOT GENERATED')}")
    print(f"Target Group: {test_data.get('target_group')}")
    print(f"Amount: {test_data.get('amount')}")
    print(f"Up To: {test_data.get('up_to')}")
    print(f"Optional: {test_data.get('optional')}")
    
    if 'output_value_key' in test_data:
        print("\n✓ SUCCESS: DISCARD command will output discarded card IDs to variable:", test_data['output_value_key'])
    else:
        print("\n❌ FAILURE: output_value_key was not generated")
    
    print("\n=== Usage Example ===")
    print("When this DISCARD command executes, it will store:")
    print("  - Discarded card instance IDs")
    print("  - Count of cards discarded")
    print(f"  Into variable: {test_data.get('output_value_key', 'var_DISCARD_N')}")
    print("\nLater commands can reference this variable via input_value_key.")
    
    app.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())
