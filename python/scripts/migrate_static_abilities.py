import json
import os
import sys

# Add bin path for potential imports
sys.path.append('bin')

def migrate_static_abilities(filepath):
    print(f"Migrating {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cards = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    migrated_count = 0

    for card in cards:
        new_effects = []
        static_abilities = card.get('static_abilities', [])

        # Check effects list
        if 'effects' in card:
            effects_list = card['effects']
        elif 'triggers' in card:
            effects_list = card['triggers']
        else:
            effects_list = []

        for eff in effects_list:
            trigger = eff.get('trigger', 'NONE')

            if trigger == 'PASSIVE_CONST':
                # Convert to Static Ability (ModifierDef)

                # Check for actions wrapping the modifier properties
                actions = eff.get('actions', [])
                modifier = {}
                modifier['type'] = 'NONE'

                if actions:
                    action = actions[0]
                    act_type = action.get('type', 'NONE')

                    if act_type in ["COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD"]:
                        modifier['type'] = act_type
                    elif act_type == "APPLY_MODIFIER":
                        pass

                    # Heuristic for implicit types in legacy data (Speed Attacker etc)
                    if modifier['type'] == 'NONE':
                        str_val = action.get('str_val', '')
                        if str_val:
                            modifier['type'] = 'GRANT_KEYWORD'
                        elif action.get('value1', 0) != 0:
                            # Guessing Power or Cost? Usually explicit.
                            pass

                    # Map values
                    modifier['value'] = action.get('value1', 0)
                    modifier['str_val'] = action.get('str_val', '')
                    modifier['filter'] = action.get('filter', {})

                else:
                    # Legacy Schema without actions
                    if 'layer_type' in eff:
                        modifier['type'] = eff['layer_type']
                    elif 'type' in eff:
                        modifier['type'] = eff['type']

                    if modifier['type'] == 'NONE' or not modifier['type']:
                         # Fallback inference
                         str_val = eff.get('str_val', eff.get('layer_str', ''))
                         if str_val:
                              modifier['type'] = 'GRANT_KEYWORD'

                    modifier['value'] = eff.get('value', eff.get('layer_value', 0))
                    modifier['str_val'] = eff.get('str_val', eff.get('layer_str', ''))

                    if 'filter' in eff:
                        modifier['filter'] = eff['filter']

                # Final check if type is still NONE but we have data
                if modifier['type'] == 'NONE' and modifier.get('str_val'):
                     modifier['type'] = 'GRANT_KEYWORD'

                # Condition
                cond = eff.get('condition', eff.get('trigger_condition', {}))
                modifier['condition'] = cond

                # If filter is missing, default to empty
                if 'filter' not in modifier:
                     modifier['filter'] = {}

                static_abilities.append(modifier)
                migrated_count += 1
            else:
                new_effects.append(eff)

        # Update card
        if 'effects' in card:
            card['effects'] = new_effects
        if 'triggers' in card:
            card['triggers'] = new_effects

        if static_abilities:
            card['static_abilities'] = static_abilities

    # Save
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=4, ensure_ascii=False)

    print(f"Migration complete. Migrated {migrated_count} effects.")

if __name__ == "__main__":
    migrate_static_abilities("data/cards.json")
