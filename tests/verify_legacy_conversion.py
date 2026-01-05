
import sys
import os
import json

# Add bin path to find dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built.")
    sys.exit(1)

def verify_conversion():
    # Helper to get all EffectPrimitive values
    primitives = [
        dm_ai_module.EffectPrimitive.DRAW_CARD,
        dm_ai_module.EffectPrimitive.ADD_MANA,
        dm_ai_module.EffectPrimitive.DESTROY,
        dm_ai_module.EffectPrimitive.RETURN_TO_HAND,
        dm_ai_module.EffectPrimitive.SEND_TO_MANA,
        dm_ai_module.EffectPrimitive.TAP,
        dm_ai_module.EffectPrimitive.UNTAP,
        dm_ai_module.EffectPrimitive.MODIFY_POWER,
        dm_ai_module.EffectPrimitive.BREAK_SHIELD,
        dm_ai_module.EffectPrimitive.LOOK_AND_ADD,
        dm_ai_module.EffectPrimitive.SUMMON_TOKEN,
        dm_ai_module.EffectPrimitive.SEARCH_DECK_BOTTOM,
        dm_ai_module.EffectPrimitive.MEKRAID,
        dm_ai_module.EffectPrimitive.DISCARD,
        dm_ai_module.EffectPrimitive.PLAY_FROM_ZONE,
        dm_ai_module.EffectPrimitive.COST_REFERENCE,
        dm_ai_module.EffectPrimitive.LOOK_TO_BUFFER,
        dm_ai_module.EffectPrimitive.SELECT_FROM_BUFFER,
        dm_ai_module.EffectPrimitive.PLAY_FROM_BUFFER,
        dm_ai_module.EffectPrimitive.MOVE_BUFFER_TO_ZONE,
        dm_ai_module.EffectPrimitive.REVOLUTION_CHANGE,
        dm_ai_module.EffectPrimitive.COUNT_CARDS,
        dm_ai_module.EffectPrimitive.GET_GAME_STAT,
        dm_ai_module.EffectPrimitive.APPLY_MODIFIER,
        dm_ai_module.EffectPrimitive.REVEAL_CARDS,
        dm_ai_module.EffectPrimitive.REGISTER_DELAYED_EFFECT,
        dm_ai_module.EffectPrimitive.RESET_INSTANCE,
        dm_ai_module.EffectPrimitive.SEARCH_DECK,
        dm_ai_module.EffectPrimitive.SHUFFLE_DECK,
        dm_ai_module.EffectPrimitive.ADD_SHIELD,
        dm_ai_module.EffectPrimitive.SEND_SHIELD_TO_GRAVE,
        dm_ai_module.EffectPrimitive.SEND_TO_DECK_BOTTOM,
        dm_ai_module.EffectPrimitive.MOVE_TO_UNDER_CARD,
        dm_ai_module.EffectPrimitive.SELECT_NUMBER,
        dm_ai_module.EffectPrimitive.FRIEND_BURST,
        dm_ai_module.EffectPrimitive.GRANT_KEYWORD,
        dm_ai_module.EffectPrimitive.MOVE_CARD,
        dm_ai_module.EffectPrimitive.CAST_SPELL,
        dm_ai_module.EffectPrimitive.PUT_CREATURE,
        dm_ai_module.EffectPrimitive.SELECT_OPTION,
        dm_ai_module.EffectPrimitive.RESOLVE_BATTLE
    ]

    # We will use JsonLoader to load a dummy card for each primitive
    # and check if the command type is set correctly (not NONE)

    missing_conversions = []

    # Use IDs within uint16_t range (0-65535)
    base_id = 30000

    for prim in primitives:
        # Create a dummy JSON for a card with this primitive
        prim_str = str(prim).split('.')[-1] # e.g. EffectPrimitive.DRAW_CARD -> DRAW_CARD

        current_id = base_id + int(prim)

        card_json = {
            "id": current_id,
            "name": f"Test-{prim_str}",
            "civilization": "FIRE",
            "type": "CREATURE",
            "cost": 1,
            "power": 1000,
            "races": ["Human"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "actions": [
                        {
                            "type": prim_str,
                            "value1": 1
                        }
                    ]
                }
            ]
        }

        # Save to temp file
        filename = f"temp_card_{prim_str}.json"
        with open(filename, 'w') as f:
            json.dump([card_json], f)

        try:
            # Load
            card_map = dm_ai_module.JsonLoader.load_cards(filename)

            # Retrieve using correct ID
            # Ensure ID is treated as int for binding
            card_def = card_map[int(current_id)]

            # Check commands
            if len(card_def.effects) > 0:
                effect = card_def.effects[0]
                if len(effect.commands) == 0:
                     missing_conversions.append(prim_str)
                else:
                    cmd = effect.commands[0]
                    if cmd.type == dm_ai_module.CommandType.NONE:
                        missing_conversions.append(f"{prim_str} (Mapped to NONE)")
            else:
                missing_conversions.append(f"{prim_str} (No Effect Loaded)")

        except Exception as e:
            print(f"Error checking {prim_str}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    if missing_conversions:
        print("Missing Conversions detected:")
        for m in missing_conversions:
            print(f" - {m}")
        sys.exit(1)
    else:
        print("All legacy primitives converted successfully!")
        sys.exit(0)

if __name__ == "__main__":
    verify_conversion()
