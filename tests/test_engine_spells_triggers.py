import sys
import os
import pytest
import time

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

class TestEngineSpellsTriggers:
    @classmethod
    def setup_class(cls):
        # Load real cards if available, otherwise mock
        try:
            cls.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
            print(f"Loaded {len(cls.card_db)} cards.")
        except Exception as e:
            print(f"Failed to load cards: {e}")
            cls.card_db = {}

    def test_play_spell(self):
        """Test playing a spell card."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Find a spell
        spell_id = -1
        for cid, card in self.card_db.items():
            if card.type == dm_ai_module.CardType.SPELL and card.cost <= 3:
                spell_id = cid
                break

        if spell_id == -1:
            pytest.skip("No suitable spell found for test")

        config = dm_ai_module.ScenarioConfig()
        config.my_hand_cards = [spell_id]
        config.my_mana_zone = [spell_id] * 5 # Plenty of mana
        gi.reset_with_scenario(config)

        # Ensure phase is MAIN
        state = gi.state
        state.current_phase = dm_ai_module.Phase.MAIN

        p0 = state.players[0]

        # Generate legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        play_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.PLAY_CARD and action.card_id == spell_id:
                play_action = action
                break

        assert play_action is not None, "Play spell action should be available"

        # Resolve action
        dm_ai_module.EffectResolver.resolve_action(state, play_action, self.card_db)

        # Verify spell is in graveyard (after resolution)
        # Note: In DM, spells go to graveyard after effect.
        # If the spell has complex targeting, it might be in pending state, but basic spells should resolve.
        assert len(p0.graveyard) == 1
        assert p0.graveyard[0].card_id == spell_id
        assert len(p0.hand) == 0

    def test_shield_trigger(self):
        """Test shield trigger activation logic."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Find a Shield Trigger card
        trigger_id = -1
        for cid, card in self.card_db.items():
            # Check keywords struct for shield_trigger
            # Note: access pattern depends on binding. Assuming card.keywords.shield_trigger
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

        # Resolve Pass -> Battle Execution -> Shield Break -> Trigger Check
        dm_ai_module.EffectResolver.resolve_action(state, pass_action, self.card_db)

        # Now, if a shield trigger was broken, the game should offer USE_SHIELD_TRIGGER action.
        # This usually happens immediately if the engine handles it, or pending effects are populated.
        # Let's check legal actions for P0.

        # Note: Turn player is still P1 technically during the attack resolution?
        # Or does priority shift to P0 to use trigger?
        # Usually in DM engine implementation, if a trigger is pending, the owner of the trigger gets priority.

        trigger_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

        use_trigger_action = None
        for action in trigger_actions:
            if action.type == dm_ai_module.ActionType.USE_SHIELD_TRIGGER:
                use_trigger_action = action
                break

        # If the engine correctly identified the trigger, this action should exist.
        assert use_trigger_action is not None, "Should have option to use shield trigger"
        # card_id is not populated in USE_SHIELD_TRIGGER in action_generator.cpp, it uses source_instance_id
        # We can verify it by checking if it matches the pending effect source instance id, but that's hard from here.
        # Instead, just assume if the action type is correct, it's the right action for now.

        # Resolve Trigger
        dm_ai_module.EffectResolver.resolve_action(state, use_trigger_action, self.card_db)

        # After using trigger, the spell effect resolves (goes to graveyard) or creature enters battle zone.
        # Check where the card went.
        p0 = state.players[0]

        card_in_grave = any(c.card_id == trigger_id for c in p0.graveyard)
        card_in_battle = any(c.card_id == trigger_id for c in p0.battle_zone)

        assert card_in_grave or card_in_battle, "Trigger card should be played (grave or battle zone)"
