
import sys
import os
import json
import pytest

# Add bin path for dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), "../bin"))
try:
    import dm_ai_module
except ImportError:
    pass

class TestHyperEnergy:
    def create_json_file(self, filename, cards):
        with open(filename, 'w') as f:
            json.dump(cards, f)

    def populate_card_db(self, card_db, cards_json):
        for c in cards_json:
            cd = card_db[c['id']]
            cd.id = c['id']
            cd.name = c['name']
            cd.cost = c['cost']
            cd.power = c.get('power', 0)
            cd.civilization = dm_ai_module.Civilization.FIRE
            cd.type = dm_ai_module.CardType.CREATURE
            cd.races = c.get('races', [])

            # Manually set keyword for test environment if not loaded by JsonLoader correctly in all contexts
            # But JsonLoader.load_cards should handle it.
            # We must sync the card_db passed to start_game though.

            # Note: The C++ JsonLoader sets the global registry or similar,
            # but start_game takes a card_db map. We need to manually set flags if we construct card_db from scratch.

            # Check for hyper energy keyword in JSON
            # In C++, JsonLoader handles this: if (action.str_val == "HYPER_ENERGY") def.keywords.hyper_energy = true;
            # Here we must mimic it or rely on loading.

            # Since we pass card_db to start_game, we need to set it here.
            for eff in c.get('effects', []):
                for act in eff.get('actions', []):
                    # Check for str_val or value mapping
                    if act.get('type') == 'COST_REFERENCE' and (act.get('str_val') == 'HYPER_ENERGY' or act.get('value') == 'HYPER_ENERGY'):
                        cd.keywords.hyper_energy = True

    def test_hyper_energy_flow(self):
        # Card 1000: Hyper Energy Creature, Cost 6
        cards_json = [
            {
                "id": 1000, "name": "Hyper Creature", "cost": 6, "civilization": "FIRE", "type": "CREATURE", "power": 6000, "races": [],
                "effects": [{
                    "trigger": "NONE", # Passive/Keyword usually handled differently but lets put it here
                    "condition": {"type": "NONE"},
                    "actions": [{
                        "type": "COST_REFERENCE",
                        "str_val": "HYPER_ENERGY"
                    }]
                }]
            },
            {
                "id": 1, "name": "Dummy", "cost": 2, "civilization": "FIRE", "type": "CREATURE", "power": 1000, "races": [],
                "effects": []
            }
        ]

        json_path = "test_cards_hyper.json"
        self.create_json_file(json_path, cards_json)
        dm_ai_module.JsonLoader.load_cards(json_path)

        game = dm_ai_module.GameInstance()
        card_db_defs = {c['id']: dm_ai_module.CardDefinition() for c in cards_json}
        self.populate_card_db(card_db_defs, cards_json)

        game.start_game(card_db_defs)
        state = game.state
        pid = state.active_player_id

        # Verify card definition
        print(f"Checking card {1000} hyper_energy: {card_db_defs[1000].keywords.hyper_energy}")

        # Setup:
        # Hand: Hyper Creature (ID 1000)
        # Battle Zone: 2 Dummies (ID 1) - Untapped
        # Mana Zone: 2 Cards (Cost 6 - (2*2) = 2. So need 2 mana)

        state.set_deck(pid, [])
        state.add_card_to_hand(pid, 1000, 1000) # Instance 1000

        # Add 2 creatures to battle zone
        # Important: Creatures must NOT have summoning sickness to be used for Hyper Energy (based on C++ impl)!
        # Use add_test_card_to_battle for detailed control

        state.add_test_card_to_battle(pid, 1, 101, False, False) # ID 1, Inst 101, Tapped=False, Sick=False
        state.add_test_card_to_battle(pid, 1, 102, False, False) # ID 1, Inst 102, Tapped=False, Sick=False

        # Add 2 mana
        state.add_card_to_mana(pid, 1, 201)
        state.add_card_to_mana(pid, 1, 202)

        # Ensure Phase is MAIN
        state.current_phase = dm_ai_module.Phase.MAIN

        # Step 1: Generate Actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db_defs)

        # Look for Hyper Energy Play Action
        # Should be PLAY_CARD with target_player=254 and target_slot_index=2 (Tap 2)
        hyper_action = None
        for a in actions:
            if a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 1000:
                print(f"Action: {a.type}, pid={a.target_player}, slot={a.target_slot_index}")
                if a.target_player == 254 and a.target_slot_index == 2:
                    hyper_action = a
                    break

        assert hyper_action is not None, "Should generate Hyper Energy action tapping 2 creatures"

        # Step 2: Resolve Play Action
        dm_ai_module.EffectResolver.resolve_action(state, hyper_action, card_db_defs)

        # Expect Pending Effect for Target Selection
        # Use helper since pending_effects is not directly exposed as list of objects but via helper tuple list
        pe_info = dm_ai_module.get_pending_effects_verbose(state)
        # tuple: (type, source_instance_id, controller, num_targets_needed, current_targets_count, has_effect_def)

        assert len(pe_info) == 1
        pe = pe_info[0]
        # ResolveType is not in the tuple, but num_targets_needed is
        print(f"Pending Effect: {pe}")
        # pe[3] is num_targets_needed
        assert pe[3] == 2

        # Step 3: Generate Selection Actions
        actions_sel = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db_defs)

        # Should have SELECT_TARGET actions for instance 101 and 102
        sel_101 = None
        sel_102 = None
        for a in actions_sel:
            if a.type == dm_ai_module.ActionType.SELECT_TARGET:
                if a.target_instance_id == 101: sel_101 = a
                if a.target_instance_id == 102: sel_102 = a

        assert sel_101 is not None
        assert sel_102 is not None

        # Step 4: Select Targets
        dm_ai_module.EffectResolver.resolve_action(state, sel_101, card_db_defs)
        dm_ai_module.EffectResolver.resolve_action(state, sel_102, card_db_defs)

        # Step 5: Resolve Effect (Finish Hyper Energy)
        # After selecting 2 targets (which matches num_targets_needed),
        # ActionGenerator should produce RESOLVE_EFFECT or we can call it if the queue is ready.
        # Actually ActionGenerator produces RESOLVE_EFFECT when ready.

        actions_res = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db_defs)
        resolve_act = None
        for a in actions_res:
            if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT:
                resolve_act = a
                break

        assert resolve_act is not None, "Should be ready to resolve effect"

        dm_ai_module.EffectResolver.resolve_action(state, resolve_act, card_db_defs)

        # Verification:
        # 1. Hyper Creature (1000) in Battle Zone
        player = state.players[pid]
        in_bz = False
        for c in player.battle_zone:
            if c.instance_id == 1000: in_bz = True
        assert in_bz, "Hyper Creature should be in Battle Zone"

        # 2. Dummies (101, 102) Tapped
        tapped_count = 0
        for c in player.battle_zone:
            if c.instance_id in [101, 102] and c.is_tapped:
                tapped_count += 1
        assert tapped_count == 2, "Both creatures should be tapped"

        # 3. Mana used (2 mana tapped)
        mana_tapped = 0
        for c in player.mana_zone:
            if c.is_tapped: mana_tapped += 1
        assert mana_tapped == 2, "2 Mana should be used"

        os.remove(json_path)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
