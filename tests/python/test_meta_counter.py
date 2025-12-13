
import sys
import os
import unittest
import pytest

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

try:
    import dm_ai_module
    from dm_ai_module import GameState, ActionGenerator, EffectResolver, PhaseManager, Action, ActionType, EffectType, Phase, SpawnSource, CardData, GameResult, CardDefinition, Civilization, CardType
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

class TestMetaCounter(unittest.TestCase):
    def setUp(self):
        self.game = GameState(1000)
        self.card_db = {}

        # Define Meta Counter Card (ID 100)
        self.meta_card_id = 100
        self.meta_card_def = CardDefinition()
        self.meta_card_def.name = "Meta Counter Unit"
        self.meta_card_def.cost = 8
        self.meta_card_def.civilizations = [Civilization.FIRE]
        self.meta_card_def.type = CardType.CREATURE
        self.meta_card_def.power = 6000
        self.meta_card_def.keywords.meta_counter_play = True
        self.card_db[self.meta_card_id] = self.meta_card_def

        # Define 0-cost Card (ID 99) for opponent to play
        self.zero_card_id = 99
        self.zero_card_def = CardDefinition()
        self.zero_card_def.name = "Zero Cost Unit"
        self.zero_card_def.cost = 0
        self.zero_card_def.civilizations = [Civilization.LIGHT]
        self.zero_card_def.type = CardType.CREATURE
        self.zero_card_def.power = 1000
        self.card_db[self.zero_card_id] = self.zero_card_def

        # Register for engine
        # We need to register CardData for EffectResolver to work properly if it uses registry,
        # but here we pass card_db explicitly to resolve_action.
        # However, PhaseManager might need it.
        # Let's register dummy data.
        cdata_meta = CardData(self.meta_card_id, "Meta Counter Unit", 8, "FIRE", 6000, "CREATURE", [], [])
        # Set keyword in CardData if possible? No direct binding for setting keywords dict on CardData in python easily
        # except via constructor or specialized setter if exposed.
        # For now, we rely on the card_db passed to resolve_action/generate_legal_actions.
        dm_ai_module.register_card_data(cdata_meta)

        cdata_zero = CardData(self.zero_card_id, "Zero Cost Unit", 0, "LIGHT", 1000, "CREATURE", [], [])
        dm_ai_module.register_card_data(cdata_zero)


    def test_meta_counter_trigger_and_resolution(self):
        # 1. Setup Player 0
        self.game.active_player_id = 0
        self.game.current_phase = Phase.MAIN

        p0 = self.game.players[0]
        # Use add_card_to_hand(player_id, card_id, instance_id)
        if hasattr(self.game, "add_card_to_hand"):
            self.game.add_card_to_hand(0, self.zero_card_id, 1) # p0, id, inst_id

        # 2. Setup Player 1
        p1 = self.game.players[1]
        if hasattr(self.game, "add_card_to_hand"):
            self.game.add_card_to_hand(1, self.meta_card_id, 2) # p1, id, inst_id

        # 3. Player 0 plays 0-cost card
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)
        play_action = next((a for a in actions if a.type == ActionType.DECLARE_PLAY and a.card_id == self.zero_card_id), None)

        # Let's add a mana of Light just in case.
        if hasattr(self.game, "add_card_to_mana"):
             self.game.add_card_to_mana(0, self.zero_card_id, 3) # p0, id, inst_id
             p0.mana_zone[0].is_tapped = False

        # Re-generate actions
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)
        play_action = next((a for a in actions if a.type == ActionType.DECLARE_PLAY and a.card_id == self.zero_card_id), None)
        self.assertIsNotNone(play_action, "Should be able to play 0 cost card")

        # Execute Play
        EffectResolver.resolve_action(self.game, play_action, self.card_db)

        # Check Stack
        self.assertEqual(len(self.game.stack_zone), 1)

        # PAY_COST
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)
        pay_action = next((a for a in actions if a.type == ActionType.PAY_COST), None)
        self.assertIsNotNone(pay_action)
        EffectResolver.resolve_action(self.game, pay_action, self.card_db)

        # RESOLVE_PLAY
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)
        resolve_action = next((a for a in actions if a.type == ActionType.RESOLVE_PLAY), None)
        self.assertIsNotNone(resolve_action)
        EffectResolver.resolve_action(self.game, resolve_action, self.card_db)

        # Verify Stats
        self.assertTrue(self.game.turn_stats.played_without_mana)

        # 4. Advance to ATTACK Phase
        self.game.current_phase = Phase.ATTACK

        # 5. Advance to END_OF_TURN
        PhaseManager.next_phase(self.game, self.card_db)

        # 6. Verify Pending Effect
        pe_info = dm_ai_module.get_pending_effects_info(self.game)
        self.assertEqual(len(pe_info), 1, "Should have pending META_COUNTER")
        pe_type, pe_source, pe_controller = pe_info[0]
        self.assertEqual(pe_type, EffectType.META_COUNTER)
        self.assertEqual(pe_controller, 1)

        # 7. Generate Actions -> Should have DECLARE_PLAY for meta counter
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)
        # Look for PLAY_CARD (or DECLARE_PLAY) for meta card (ID 100)
        # Meta Counter usually generates PLAY_CARD directly or INTERNAL_PLAY?
        # The logic says it generates actions based on PendingEffect.
        # If it's a Hand Trigger (Meta Counter), it allows playing the card.

        # ActionGenerator should generate a PLAY action for the pending effect
        # Actually, for META_COUNTER, it might be USE_ABILITY or similar?
        # Let's check ActionGenerator logic or just look for any action for card 100.

        print("Actions generated:")
        for a in actions:
            print(f"Type: {a.type}, CardID: {a.card_id}, SourceInst: {a.source_instance_id}")

        meta_action = next((a for a in actions if a.card_id == self.meta_card_id), None)
        # The action type depends on implementation (likely PLAY_CARD or USE_ABILITY)
        self.assertIsNotNone(meta_action)
