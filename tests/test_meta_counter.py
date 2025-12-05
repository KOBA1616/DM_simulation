
import unittest
import dm_ai_module
from dm_ai_module import GameState, ActionGenerator, EffectResolver, PhaseManager, Action, ActionType, EffectType, Phase, SpawnSource, CardData, GameResult, CardDefinition, Civilization, CardType

class TestMetaCounter(unittest.TestCase):
    def setUp(self):
        self.game = GameState(1000)
        self.card_db = {}

        # Define Meta Counter Card (ID 100)
        self.meta_card_id = 100
        self.meta_card_def = CardDefinition()
        self.meta_card_def.name = "Meta Counter Unit"
        self.meta_card_def.cost = 8
        self.meta_card_def.civilization = Civilization.FIRE
        self.meta_card_def.type = CardType.CREATURE
        self.meta_card_def.power = 6000
        self.meta_card_def.keywords.meta_counter_play = True
        self.meta_card_def.keywords.speed_attacker = True
        self.card_db[self.meta_card_id] = self.meta_card_def

        # Define Zero Cost Play Card (ID 200)
        self.zero_card_id = 200
        self.zero_card_def = CardDefinition()
        self.zero_card_def.name = "Zero Cost Spell"
        self.zero_card_def.cost = 0 # Base cost 0
        self.zero_card_def.civilization = Civilization.LIGHT
        self.zero_card_def.type = CardType.SPELL
        self.card_db[self.zero_card_id] = self.zero_card_def

        # Initialize Game
        PhaseManager.start_game(self.game, self.card_db)

        # Clear hands/zones for control
        self.game.players[0].hand.clear()
        self.game.players[0].mana_zone.clear()
        self.game.players[0].battle_zone.clear()
        self.game.players[1].hand.clear()
        self.game.players[1].mana_zone.clear()
        self.game.players[1].battle_zone.clear()

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
        # PendingEffects are not directly exposed as a list property on GameState in Python bindings (my memory was wrong or incomplete)
        # Use get_pending_effects_info helper
        pe_info = dm_ai_module.get_pending_effects_info(self.game)
        self.assertEqual(len(pe_info), 1, "Should have pending META_COUNTER")
        # tuple: (type, source_instance_id, controller)
        pe_type, pe_source, pe_controller = pe_info[0]
        self.assertEqual(pe_type, int(EffectType.META_COUNTER))
        self.assertEqual(pe_controller, 1)

        # 7. Verify Action Generation
        actions = ActionGenerator.generate_legal_actions(self.game, self.card_db)

        internal_play = next((a for a in actions if a.type == ActionType.PLAY_CARD_INTERNAL), None)
        self.assertIsNotNone(internal_play)
        self.assertEqual(internal_play.source_instance_id, 2) # instance_id we set earlier
        self.assertEqual(internal_play.spawn_source, SpawnSource.HAND_SUMMON)

        # 8. Resolve Meta Counter Play
        EffectResolver.resolve_action(self.game, internal_play, self.card_db)

        # Check Battle Zone
        self.assertEqual(len(p1.battle_zone), 1)
        self.assertEqual(p1.battle_zone[0].card_id, self.meta_card_id)

        # Check Pending Effect removed
        pe_info_final = dm_ai_module.get_pending_effects_info(self.game)
        self.assertEqual(len(pe_info_final), 0)

if __name__ == '__main__':
    unittest.main()
