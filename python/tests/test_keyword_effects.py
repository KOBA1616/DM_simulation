
import pytest
import dm_ai_module

def test_mega_last_burst_effect_generation():
    """Verify Mega Last Burst creates ON_DESTROY effect with CAST_SPELL (SELF)"""

    # Create a minimal JSON snippet with Mega Last Burst
    json_data = """
    {
        "id": 1000,
        "name": "MLB Test",
        "type": "CREATURE",
        "keywords": {
            "mega_last_burst": true
        },
        "spell_side": {
            "id": 1001,
            "name": "MLB Spell",
            "type": "SPELL",
            "effects": [
                {
                    "trigger": "NONE",
                    "actions": [
                        {
                            "type": "ADD_MANA",
                            "value1": 1
                        }
                    ]
                }
            ]
        }
    }
    """

    # Load via CardRegistry (which calls JsonLoader)
    dm_ai_module.CardRegistry.load_from_json(json_data)

    # Access definitions via registry or db
    # We can use card_registry_get_all_definitions but it returns a dict
    # Wait, binding might be get_all_definitions or something
    # Let's check bindings. In python/tests, we can inspect module.

    # Assuming standard usage, we load and then check.
    # The actual loading logic logic is inside load_cards mostly, but load_from_json uses similar helpers.
    # However, JsonLoader::load_cards calls CardRegistry::load_from_json, AND does convert_to_def.
    # CardRegistry::load_from_json updates the registry.
    # We need to see if the registry definition has the effect.

    # Re-reading JsonLoader.cpp:
    # load_cards:
    #   CardRegistry::load_from_json(item.dump());
    #   result[...] = convert_to_def(card);

    # CardRegistry stores CardData (raw struct).
    # But Engine uses CardDefinition.
    # The convert_to_def logic is what we changed.
    # We need to call JsonLoader.load_cards or equivalent if exposed.
    # dm_ai_module.JsonLoader.load_cards returns a dict[int, CardDefinition].

    # So we should write to a temp file and load it.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write("[" + json_data + "]")
        tmp_path = tmp.name

    try:
        defs = dm_ai_module.JsonLoader.load_cards(tmp_path)
        card_def = defs[1000]

        # Check effects
        assert card_def.keywords.mega_last_burst == True

        has_mlb_effect = False
        for eff in card_def.effects:
            if eff.trigger == dm_ai_module.TriggerType.ON_DESTROY:
                for act in eff.actions:
                    if act.type == dm_ai_module.EffectPrimitive.CAST_SPELL:
                        if act.cast_spell_side == True:
                            # Scope check if exposed, or implied
                            has_mlb_effect = True

                            # Verify scope is SELF (if exposed)
                            if hasattr(act, 'scope'):
                                assert act.scope == dm_ai_module.TargetScope.SELF

        assert has_mlb_effect, "Mega Last Burst effect not found in CardDefinition"

    finally:
        os.remove(tmp_path)

def test_friend_burst_effect_generation():
    """Verify Friend Burst creates ON_PLAY effect"""
    json_data = """
    {
        "id": 2000,
        "name": "FB Test",
        "type": "CREATURE",
        "keywords": {
            "friend_burst": true
        }
    }
    """

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write("[" + json_data + "]")
        tmp_path = tmp.name

    try:
        defs = dm_ai_module.JsonLoader.load_cards(tmp_path)
        card_def = defs[2000]

        assert card_def.keywords.friend_burst == True

        has_fb_effect = False
        for eff in card_def.effects:
            if eff.trigger == dm_ai_module.TriggerType.ON_PLAY:
                for act in eff.actions:
                    if act.type == dm_ai_module.EffectPrimitive.FRIEND_BURST:
                        has_fb_effect = True
                        if hasattr(act, 'filter'):
                            assert act.filter.owner == "SELF"
                            assert "BATTLE_ZONE" in act.filter.zones

        assert has_fb_effect, "Friend Burst effect not found in CardDefinition"

    finally:
        os.remove(tmp_path)
