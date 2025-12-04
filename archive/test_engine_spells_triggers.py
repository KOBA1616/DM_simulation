import sys
import os
import pytest
import time
import json

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

class TestEngineSpellsTriggers:
    @classmethod
    def setup_class(cls):
        # Use JsonLoader as CsvLoader is legacy/deprecated
        try:
            cls.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
            print(f"Loaded {len(cls.card_db)} cards.")
        except Exception as e:
            print(f"Failed to load cards: {e}")
            cls.card_db = {}

    def test_play_spell(self):
        """Test playing a spell card."""
        # Use a temporary custom spell to ensure it has valid effects and correct setup
        spell_id = 999
        cards_json = [
             {
                "id": spell_id, "name": "Test Spell", "type": "SPELL", "civilization": "FIRE", "cost": 1,
                "effects": [{"trigger": "ON_PLAY", "actions": [{"type": "ADD_MANA", "scope": "PLAYER_SELF", "value1": 1}]}]
             }
        ]
        temp_json = "test_cards_spell_temp.json"
        with open(temp_json, "w") as f:
            json.dump(cards_json, f)

        card_db = dm_ai_module.JsonLoader.load_cards(temp_json)
        gi = dm_ai_module.GameInstance(42, card_db)

        config = dm_ai_module.ScenarioConfig()
        config.my_hand_cards = [spell_id]
        config.my_mana_zone = [spell_id] * 2 # Plenty of mana
        gi.reset_with_scenario(config)

        # Ensure phase is MAIN
        state = gi.state
        state.current_phase = dm_ai_module.Phase.MAIN

        p0 = state.players[0]

        # Generate legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
        play_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.PLAY_CARD and action.card_id == spell_id:
                play_action = action
                break

        assert play_action is not None, "Play spell action should be available"

        # Resolve action
        dm_ai_module.EffectResolver.resolve_action(state, play_action, card_db)

        # Process any pending effects (ADD_MANA)
        for _ in range(5):
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
            resolve_act = None
            for a in actions:
                if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT:
                    resolve_act = a
                    break

            if resolve_act:
                 dm_ai_module.EffectResolver.resolve_action(state, resolve_act, card_db)
            else:
                break

        # Verify spell is in graveyard (after resolution)
        assert len(p0.graveyard) == 1
        assert p0.graveyard[0].card_id == spell_id
        assert len(p0.hand) == 0

        # Verify effect happened (mana added)
        # We started with 2 mana, used 1, added 1 -> 2 mana.
        # But wait, we used 1 mana (tapped). Effect adds 1 mana (untapped usually or tapped?).
        # Mana zone count should be 3? (2 initial + 1 added).
        # ADD_MANA action puts top of deck to mana.
        # But scenario initializes deck empty?
        # reset_with_scenario sets deck to 30 dummy cards (ID 1) usually.
        # So it should work.
        assert len(p0.mana_zone) == 3

        os.remove(temp_json)

    def test_shield_trigger(self):
        """Test shield trigger activation logic."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Find a Shield Trigger card
        trigger_id = -1
        for cid, card in self.card_db.items():
            # Check keywords struct for shield_trigger
            if card.keywords.shield_trigger:
                trigger_id = cid
                break

        if trigger_id == -1:
            pytest.skip("No shield trigger card found")

        # Find an attacker for opponent
        attacker_id = -1
        for cid, card in self.card_db.items():
            if card.type == dm_ai_module.CardType.CREATURE:
                attacker_id = cid
                break

        if attacker_id == -1:
            pytest.skip("No attacker found")

        config = dm_ai_module.ScenarioConfig()
        config.my_shields = [trigger_id] # Player 0 has trigger in shield
        config.enemy_battle_zone = [attacker_id] # Player 1 has attacker
        gi.reset_with_scenario(config)

        state = gi.state

        # Force turn to Player 1 (Opponent) to simulate their attack
        state.active_player_id = 1
        state.current_phase = dm_ai_module.Phase.ATTACK

        # Generate actions for P1
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

        attack_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.ATTACK_PLAYER:
                attack_action = action
                break

        assert attack_action is not None, "Opponent should be able to attack"

        # Resolve Attack
        dm_ai_module.EffectResolver.resolve_action(state, attack_action, self.card_db)

        # Phase should become BLOCK (P0 can block)
        assert state.current_phase == dm_ai_module.Phase.BLOCK

        # P0 passes block
        pass_action = None
        block_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        for action in block_actions:
            if action.type == dm_ai_module.ActionType.PASS:
                pass_action = action
                break

        assert pass_action is not None

        # Resolve Pass -> queues RESOLVE_BATTLE
        dm_ai_module.EffectResolver.resolve_action(state, pass_action, self.card_db)

        # Now we must process the battle resolution queue (RESOLVE_BATTLE -> BREAK_SHIELD -> USE_SHIELD_TRIGGER)
        use_trigger_action = None

        for _ in range(10): # Safety limit
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
            if not actions: break

            # Check for trigger first
            for a in actions:
                if a.type == dm_ai_module.ActionType.USE_SHIELD_TRIGGER:
                    use_trigger_action = a
                    break
            if use_trigger_action: break

            # Check for system actions to advance
            system_action = None
            for a in actions:
                if a.type in [dm_ai_module.ActionType.RESOLVE_BATTLE, dm_ai_module.ActionType.BREAK_SHIELD]:
                    system_action = a
                    break

            if system_action:
                dm_ai_module.EffectResolver.resolve_action(state, system_action, self.card_db)
            else:
                break

        # If the engine correctly identified the trigger, this action should exist.
        assert use_trigger_action is not None, "Should have option to use shield trigger"

        # Resolve Trigger
        dm_ai_module.EffectResolver.resolve_action(state, use_trigger_action, self.card_db)

        # Process any pending effects from the trigger (e.g. if it's a spell or CIP creature)
        for _ in range(5):
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
            resolve_act = None
            for a in actions:
                if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT:
                    resolve_act = a
                    break
            if resolve_act:
                 dm_ai_module.EffectResolver.resolve_action(state, resolve_act, self.card_db)
            else:
                break

        # Check where the card went.
        p0 = state.players[0]

        card_in_grave = any(c.card_id == trigger_id for c in p0.graveyard)
        card_in_battle = any(c.card_id == trigger_id for c in p0.battle_zone)

        assert card_in_grave or card_in_battle, "Trigger card should be played (grave or battle zone)"
