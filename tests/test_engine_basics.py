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

class TestEngineBasics:
    @classmethod
    def setup_class(cls):
        # Load real cards if available, otherwise mock
        try:
            cls.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
            print(f"Loaded {len(cls.card_db)} cards.")
        except Exception as e:
            print(f"Failed to load cards: {e}")
            cls.card_db = {}

    def test_mana_charge(self):
        """Test charging mana functionality."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Setup scenario: Player has 1 card in hand
        config = dm_ai_module.ScenarioConfig()
        # Find a valid card ID
        card_id = 0
        if self.card_db:
            card_id = list(self.card_db.keys())[0]

        config.my_hand_cards = [card_id]
        gi.reset_with_scenario(config)

        state = gi.state
        # Manually set phase to MANA because reset_with_scenario defaults to MAIN
        state.current_phase = dm_ai_module.Phase.MANA

        p0 = state.players[0]

        assert len(p0.mana_zone) == 0
        assert len(p0.hand) == 1

        # Generate legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

        # Find mana charge action
        charge_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.MANA_CHARGE:
                charge_action = action
                break

        assert charge_action is not None, "Mana charge action should be available"

        # Execute action
        dm_ai_module.EffectResolver.resolve_action(state, charge_action, self.card_db)

        assert len(p0.mana_zone) == 1
        assert len(p0.hand) == 0

    def test_play_creature(self):
        """Test playing a creature."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Find a low cost creature
        creature_id = -1
        for cid, card in self.card_db.items():
            # Fix: use .type instead of .card_type
            if card.type == dm_ai_module.CardType.CREATURE and card.cost <= 2:
                creature_id = cid
                break

        if creature_id == -1:
            pytest.skip("No suitable creature found for test")

        config = dm_ai_module.ScenarioConfig()
        config.my_hand_cards = [creature_id]
        config.my_mana_zone = [creature_id] * 5 # Plenty of mana
        gi.reset_with_scenario(config)

        # Ensure phase is MAIN (default)
        state = gi.state
        state.current_phase = dm_ai_module.Phase.MAIN

        p0 = state.players[0]

        # Ensure mana is sufficient
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        play_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.PLAY_CARD and action.card_id == creature_id:
                play_action = action
                break

        assert play_action is not None, "Play card action should be available"

        # Initialize stats to check if they update (though actual update logic is in C++, we just check it runs)
        dm_ai_module.initialize_card_stats(state, self.card_db, 40)

        dm_ai_module.EffectResolver.resolve_action(state, play_action, self.card_db)

        assert len(p0.battle_zone) == 1
        assert p0.battle_zone[0].card_id == creature_id

        # Verify stats update (play_count should increase)
        stats = dm_ai_module.get_card_stats(state)
        if creature_id in stats:
             assert stats[creature_id]['play_count'] >= 1

    def test_attack_player(self):
        """Test attacking player and shield break."""
        gi = dm_ai_module.GameInstance(42, self.card_db)

        # Find a creature
        creature_id = -1
        for cid, card in self.card_db.items():
            # Fix: use .type instead of .card_type
            if card.type == dm_ai_module.CardType.CREATURE:
                creature_id = cid
                break

        if creature_id == -1:
             pytest.skip("No creature found")

        config = dm_ai_module.ScenarioConfig()
        config.my_battle_zone = [creature_id] # Creature on board
        config.enemy_shield_count = 1
        gi.reset_with_scenario(config)

        state = gi.state
        # Manually set phase to ATTACK because we want to attack
        # In reality, one would PASS in MAIN to get here, but we shortcut for unit test
        state.current_phase = dm_ai_module.Phase.ATTACK

        p0 = state.players[0]
        p1 = state.players[1]

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        attack_action = None
        for action in actions:
            if action.type == dm_ai_module.ActionType.ATTACK_PLAYER:
                attack_action = action
                break

        assert attack_action is not None, "Should be able to attack player"

        dm_ai_module.EffectResolver.resolve_action(state, attack_action, self.card_db)

        # After ATTACK_PLAYER action, the phase transitions to BLOCK
        assert state.current_phase == dm_ai_module.Phase.BLOCK

        # Generate actions for BLOCK phase (should be PASS if no blockers)
        block_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        pass_action = None
        for action in block_actions:
            if action.type == dm_ai_module.ActionType.PASS:
                pass_action = action
                break

        assert pass_action is not None, "Should be able to pass in block phase"

        # Resolve PASS in BLOCK phase -> executes battle
        dm_ai_module.EffectResolver.resolve_action(state, pass_action, self.card_db)

        # Check shield count
        # Wait, if shield break triggers a shield trigger, it might go to pending effects
        # But for basic test, we assume no trigger or trigger goes to pending
        # Shield should be removed from shield zone
        assert len(p1.shield_zone) == 0

    def test_card_stats_initialization(self):
        """Verify CardStats initialization."""
        gi = dm_ai_module.GameInstance(42, self.card_db)
        state = gi.state

        # Check if stats are initialized for a known card
        if not self.card_db:
            return

        # Initialize stats explicitly
        dm_ai_module.initialize_card_stats(state, self.card_db, 40)

        card_id = list(self.card_db.keys())[0]
        stats = dm_ai_module.get_card_stats(state)

        # Should exist but be empty/zero
        assert card_id in stats
        assert stats[card_id]['play_count'] == 0
