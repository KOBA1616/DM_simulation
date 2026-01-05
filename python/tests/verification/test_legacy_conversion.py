import pytest
import json
import dm_ai_module

@pytest.fixture
def primitives_to_test():
    return [
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

def test_verify_legacy_conversion(primitives_to_test, tmp_path):
    base_id = 30000
    card_list = []

    # Generate all card definitions
    for prim in primitives_to_test:
        prim_str = str(prim).split('.')[-1]
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
        card_list.append(card_json)

    # Save to a single temp file
    filename = tmp_path / "legacy_conversion_test_cards.json"
    filename.write_text(json.dumps(card_list))

    # Load cards
    card_map = dm_ai_module.JsonLoader.load_cards(str(filename))

    missing_conversions = []

    for prim in primitives_to_test:
        prim_str = str(prim).split('.')[-1]
        current_id = base_id + int(prim)

        if current_id not in card_map:
            missing_conversions.append(f"{prim_str} (ID {current_id} not found)")
            continue

        card_def = card_map[current_id]

        if len(card_def.effects) > 0:
            effect = card_def.effects[0]
            has_valid_conversion = False

            # Check legacy commands (deprecated but still used for some atomic actions)
            if len(effect.commands) > 0:
                 if effect.commands[0].type != dm_ai_module.CommandType.NONE:
                     has_valid_conversion = True

            # Check new ActionDef system (preferred)
            if not has_valid_conversion and hasattr(effect, 'actions'):
                if len(effect.actions) > 0:
                    # If actions are populated, it means the primitive was successfully mapped to an ActionDef
                    has_valid_conversion = True

            if not has_valid_conversion:
                missing_conversions.append(prim_str)
        else:
            missing_conversions.append(f"{prim_str} (No Effect Loaded)")

    assert not missing_conversions, f"Missing Conversions (Neither Command nor Action generated): {missing_conversions}"
