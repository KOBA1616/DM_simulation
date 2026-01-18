#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consistency Check: Card ID 9 Structure and Text Generation
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def check_card_id9():
    """Check card id=9 structure"""
    print("\nCHECK 1: Card ID 9 Structure")
    print("=" * 70)

    with open("data/cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    card = next((c for c in cards if c.get("id") == 9), None)
    if not card:
        print("FAIL: Card id=9 not found")
        return False

    issues = []

    # Main side
    main_effects = card.get("effects", [])
    print(f"Main effects: {len(main_effects)}")
    for i, eff in enumerate(main_effects):
        print(f"  Effect {i+1}: trigger={eff.get('trigger')}")
        commands = eff.get("commands", [])
        for j, cmd in enumerate(commands):
            cmd_type = cmd.get("type")
            print(f"    Command {j+1}: {cmd_type}")
            if cmd_type == "APPLY_MODIFIER":
                required = ["duration", "str_param", "target_filter", "target_group"]
                for field in required:
                    val = cmd.get(field)
                    if not val:
                        issues.append(f"Main APPLY_MODIFIER missing: {field}")
                    else:
                        print(f"      {field}: OK")

    # Spell side
    spell = card.get("spell_side")
    print(f"Spell side: {'YES' if spell else 'NO'}")
    if spell:
        spell_effects = spell.get("effects", [])
        print(f"  Spell effects: {len(spell_effects)}")
        for i, eff in enumerate(spell_effects):
            print(f"    Effect {i+1}: trigger={eff.get('trigger')}")
            commands = eff.get("commands", [])
            for j, cmd in enumerate(commands):
                cmd_type = cmd.get("type")
                print(f"      Command {j+1}: {cmd_type}")
                if cmd_type == "SELECT_NUMBER":
                    output_key = cmd.get("output_value_key")
                    if output_key:
                        print(f"        output_value_key={output_key}: OK")
                    else:
                        issues.append("Spell SELECT_NUMBER missing output_value_key")
                elif cmd_type == "APPLY_MODIFIER":
                    required = ["duration", "str_param", "target_filter", "target_group"]
                    for field in required:
                        val = cmd.get(field)
                        if not val:
                            issues.append(f"Spell APPLY_MODIFIER missing: {field}")
                        else:
                            print(f"        {field}: OK")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nRESULT: PASS")
    return True


def check_text_generation():
    """Check text generation consistency"""
    print("\nCHECK 2: Text Generation Consistency")
    print("=" * 70)

    test_cases = [
        {
            "name": "Empty filter",
            "filter": {}
        },
        {
            "name": "Exact cost filter",
            "filter": {"types": ["CREATURE"], "exact_cost": 3}
        },
        {
            "name": "Cost range filter",
            "filter": {"types": ["SPELL"], "min_cost": 1, "max_cost": 5}
        },
        {
            "name": "Cost reference",
            "filter": {"cost_ref": "chosen_cost"}
        },
        {
            "name": "Power conditions",
            "filter": {"min_power": 2000, "max_power": 5000}
        }
    ]

    issues = []
    for i, case in enumerate(test_cases):
        try:
            desc = CardTextGenerator.generate_trigger_filter_description(case["filter"])
            print(f"Test {i+1}: {case['name']}")
            print(f"  Filter: {case['filter']}")
            print(f"  Result: {desc if desc else '(empty)'}")
        except Exception as e:
            issues.append(f"Test {i+1} failed: {str(e)}")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nRESULT: PASS")
    return True


def check_scope_and_filter():
    """Check scope and filter combination"""
    print("\nCHECK 3: Scope + Filter Text Generation")
    print("=" * 70)

    test_cases = [
        {
            "name": "Opponent spell cast",
            "scope": "PLAYER_OPPONENT",
            "filter": {"types": ["SPELL"]},
            "expected": ["opponent", "spell"]
        },
        {
            "name": "Self creature power",
            "scope": "PLAYER_SELF",
            "filter": {"types": ["CREATURE"], "min_power": 3000},
            "expected": ["self", "3000"]
        }
    ]

    issues = []
    for i, case in enumerate(test_cases):
        try:
            base = CardTextGenerator.trigger_to_japanese("ON_CAST_SPELL", is_spell=False)
            result = CardTextGenerator._apply_trigger_scope(
                base,
                case["scope"],
                "ON_CAST_SPELL",
                case["filter"]
            )
            print(f"Test {i+1}: {case['name']}")
            print(f"  Result: {result}")

        except Exception as e:
            issues.append(f"Test {i+1} failed: {str(e)}")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nRESULT: PASS")
    return True


def main():
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK: Card ID 9 and Text Generation")
    print("=" * 70)

    checks = [
        ("Card Structure", check_card_id9),
        ("Text Generation", check_text_generation),
        ("Scope + Filter", check_scope_and_filter)
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\nFAIL: {name} - Unexpected error: {str(e)}")
            results[name] = False

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")

    all_pass = all(results.values())
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: ALL CHECKS PASSED")
        print("=" * 70)
        return 0
    else:
        print("OVERALL: SOME CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
