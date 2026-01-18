#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ConditionEditorWidget の破損分析
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.forms.parts.condition_widget import CONDITION_UI_CONFIG

def analyze_condition_config():
    """条件フォーム設定の破損分析"""
    print("\n" + "=" * 80)
    print("Condition Editor Widget Corruption Analysis")
    print("=" * 80)

    cond_types = [
        "NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH",
        "OPPONENT_PLAYED_WITHOUT_MANA", "OPPONENT_DRAW_COUNT",
        "DURING_YOUR_TURN", "DURING_OPPONENT_TURN",
        "FIRST_ATTACK", "EVENT_FILTER_MATCH",
        "COMPARE_STAT", "COMPARE_INPUT", "CARDS_MATCHING_FILTER", "DECK_EMPTY",
        "CUSTOM"
    ]

    issues = []

    print("\n【チェック1】Condition Type Config Coverage")
    print("-" * 80)
    for ctype in cond_types:
        has_config = ctype in CONDITION_UI_CONFIG
        label = CardTextResources.get_condition_type_label(ctype)
        
        status = "OK" if has_config else "MISSING"
        print(f"{status}: {ctype:35} -> {label} (has_config={has_config})")
        
        if not has_config:
            issues.append(f"CONDITION_UI_CONFIG missing entry for: {ctype}")

    print("\n【チェック2】populate_combo vs findData Consistency")
    print("-" * 80)
    print("When populate_combo is called:")
    print("  - Loop through cond_types")
    print("  - Get label via CardTextResources.get_condition_type_label()")
    print("  - combo.addItem(label, str(item))")
    print("    → Display: label, Data: condition_type (correct)")
    print()
    print("When set_data calls findData(ctype):")
    print("  - Searches for data value matching 'ctype'")
    print("  - Should work IF combo was properly populated with data values")
    print()
    print("Issue: If CardTextResources returns None or empty string,")
    print("       items won't display properly in combo box")

    print("\n【チェック3】CardTextResources for Condition Types")
    print("-" * 80)
    for ctype in cond_types:
        label = CardTextResources.get_condition_type_label(ctype)
        if not label or label == ctype:
            issues.append(f"CardTextResources returns empty/unmapped label for: {ctype}")
        print(f"  {ctype:35} -> {repr(label)}")

    print("\n【チェック4】CONDITION_UI_CONFIG Completeness")
    print("-" * 80)
    required_fields = ["show_val", "show_str", "label_val", "label_str"]
    for ctype, config in CONDITION_UI_CONFIG.items():
        missing = [f for f in required_fields if f not in config]
        if missing:
            issues.append(f"{ctype}: missing config fields {missing}")
        status = "OK" if not missing else "INCOMPLETE"
        print(f"{status}: {ctype:35} config={list(config.keys())}")

    print("\n" + "=" * 80)
    print("Analysis Results")
    print("=" * 80)
    
    if issues:
        print(f"\nFOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("\nNO ISSUES FOUND")
        return True


if __name__ == "__main__":
    result = analyze_condition_config()
    sys.exit(0 if result else 1)
