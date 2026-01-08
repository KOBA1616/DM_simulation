# -*- coding: utf-8 -*-
"""
Data Migration Script: Actions → Commands

既存のJSONファイルに含まれるlegacy Action形式をCommand形式に一括変換します。

使用方法:
    python scripts/migrate_actions_to_commands.py <input_json> [output_json]

- input_json: 変換元のJSONファイルパス
- output_json: 変換後のJSONファイルパス（省略時は元ファイルを上書き）
"""

import json
import sys
from pathlib import Path
from dm_toolkit.gui.editor.action_converter import ActionConverter


def migrate_effect(effect_data):
    """
    Effect内のactionsをcommandsに変換
    """
    # 既にcommandsがある場合はスキップ
    if 'commands' in effect_data and effect_data['commands']:
        print(f"  - Skipping (already has commands)")
        return effect_data, False

    actions = effect_data.get('actions', [])
    if not actions:
        return effect_data, False

    converted_commands = []
    conversion_warnings = []

    for idx, action in enumerate(actions):
        try:
            converted = ActionConverter.convert(action)
            if converted and converted.get('type') != 'NONE':
                converted_commands.append(converted)
                print(f"    ✓ Action[{idx}] {action.get('type', 'UNKNOWN')} → Command {converted.get('type', 'UNKNOWN')}")
            else:
                conversion_warnings.append(f"    ⚠ Action[{idx}] {action.get('type', 'UNKNOWN')} → NONE (skipped)")
                # Keep as warning marker
                converted_commands.append({
                    'type': 'NONE',
                    'legacy_warning': True,
                    'warning': 'Conversion resulted in NONE',
                    'legacy_original_action': action
                })
        except Exception as e:
            conversion_warnings.append(f"    ✗ Action[{idx}] {action.get('type', 'UNKNOWN')} → ERROR: {e}")
            converted_commands.append({
                'type': 'NONE',
                'legacy_warning': True,
                'warning': str(e),
                'legacy_original_action': action
            })

    # Update effect data
    effect_data['commands'] = converted_commands
    if 'actions' in effect_data:
        del effect_data['actions']

    for warning in conversion_warnings:
        print(warning)

    return effect_data, True


def migrate_card(card_data):
    """
    Card内の全Effectsを変換
    """
    modified = False

    # Effects
    effects = card_data.get('effects', [])
    for eff_idx, effect in enumerate(effects):
        print(f"  Effect[{eff_idx}] ({effect.get('trigger', 'NONE')})")
        effect, changed = migrate_effect(effect)
        if changed:
            modified = True

    # Triggers (legacy)
    triggers = card_data.get('triggers', [])
    for trig_idx, trigger in enumerate(triggers):
        print(f"  Trigger[{trig_idx}]")
        trigger, changed = migrate_effect(trigger)
        if changed:
            modified = True

    # Metamorph abilities
    metamorphs = card_data.get('metamorph_abilities', [])
    for meta_idx, meta in enumerate(metamorphs):
        print(f"  Metamorph[{meta_idx}]")
        meta, changed = migrate_effect(meta)
        if changed:
            modified = True

    # Twinpact spell side
    spell_side = card_data.get('spell_side')
    if spell_side:
        print(f"  Spell Side")
        spell_effects = spell_side.get('effects', [])
        for sp_idx, sp_eff in enumerate(spell_effects):
            print(f"    Spell Effect[{sp_idx}]")
            sp_eff, changed = migrate_effect(sp_eff)
            if changed:
                modified = True

    return card_data, modified


def migrate_file(input_path, output_path=None):
    """
    JSONファイル全体を変換
    """
    if output_path is None:
        output_path = input_path

    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_modified = 0

    # Assume data is a list of cards
    if isinstance(data, list):
        for card_idx, card in enumerate(data):
            card_name = card.get('name', f'Card {card_idx}')
            print(f"\n[{card_idx}] {card_name}")
            card, modified = migrate_card(card)
            if modified:
                total_modified += 1
    elif isinstance(data, dict):
        # Single card
        print(f"\n[Single Card] {data.get('name', 'Unknown')}")
        data, modified = migrate_card(data)
        if modified:
            total_modified += 1

    print(f"\n{'='*60}")
    print(f"Migration Complete: {total_modified} cards modified")
    print(f"Saving to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved successfully")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    migrate_file(input_file, output_file)


if __name__ == '__main__':
    main()
