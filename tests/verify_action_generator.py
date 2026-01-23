import unittest
import sys
import os

# Ensure we import the local dm_ai_module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module

class TestActionGenerator(unittest.TestCase):
    def setUp(self):
        self.game = dm_ai_module.GameInstance()
        self.state = self.game.state
        self.state.initialize()

        # Mock Card DB
        self.card_db = {
            1: {"id": 1, "name": "Test Creature", "cost": 3, "type": "CREATURE", "civilization": "FIRE"},
            2: {"id": 2, "name": "Test Spell", "cost": 2, "type": "SPELL", "civilization": "WATER"},
            3: {"id": 3, "name": "Big Creature", "cost": 99, "type": "CREATURE", "civilization": "NATURE"}
        }

    def test_generator_exists(self):
        self.assertTrue(hasattr(dm_ai_module, 'ActionGenerator'), "ActionGenerator class is missing")

    def test_mana_phase_actions(self):
        # Phase 2 = Mana
        self.state.current_phase = 2

        # Add cards to hand
        self.state.add_card_to_hand(0, 1) # Cost 3
        self.state.add_card_to_hand(0, 2) # Cost 2

        # Active player 0
        self.state.active_player_id = 0

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)

        # Expect MANA_CHARGE for each card + PASS
        mana_actions = [a for a in actions if a.type == dm_ai_module.ActionType.MANA_CHARGE]
        pass_actions = [a for a in actions if a.type == dm_ai_module.ActionType.PASS]

        self.assertEqual(len(mana_actions), 2)
        self.assertEqual(len(pass_actions), 1)

    def test_main_phase_actions(self):
        # Phase 3 = Main
        self.state.current_phase = 3
        self.state.active_player_id = 0

        # Add cards to hand
        self.state.add_card_to_hand(0, 1) # Cost 3
        self.state.add_card_to_hand(0, 3) # Cost 99

        # Setup Mana: 3 untaped cards
        self.state.add_card_to_mana(0, 1)
        self.state.add_card_to_mana(0, 1)
        self.state.add_card_to_mana(0, 1)
        # Ensure they are untapped (stub defaults to untapped usually, but let's be sure)
        for m in self.state.players[0].mana_zone:
            m.is_tapped = False

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)

        # Expect PLAY_CARD for Cost 3, but NOT for Cost 99 + PASS
        play_actions = [a for a in actions if a.type == dm_ai_module.ActionType.PLAY_CARD]
        pass_actions = [a for a in actions if a.type == dm_ai_module.ActionType.PASS]

        self.assertEqual(len(play_actions), 1, "Should only be able to play the 3-cost card")
        self.assertEqual(play_actions[0].card_id, 1)
        self.assertEqual(len(pass_actions), 1)

    def test_attack_phase_actions(self):
        # Phase 4 = Attack
        self.state.current_phase = 4
        self.state.active_player_id = 0

        # Add creature to battle zone
        c = self.state.add_test_card_to_battle(0, 1, 100, tapped=False, sick=False)
        # Add sick creature
        c2 = self.state.add_test_card_to_battle(0, 1, 101, tapped=False, sick=True)
        # Add tapped creature
        c3 = self.state.add_test_card_to_battle(0, 1, 102, tapped=True, sick=False)

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)

        # Expect ATTACK_PLAYER for c (100)
        attack_actions = [a for a in actions if a.type == dm_ai_module.ActionType.ATTACK_PLAYER]

        self.assertEqual(len(attack_actions), 1)
        self.assertEqual(attack_actions[0].source_instance_id, 100)

    def test_pending_effects(self):
        # Any phase
        self.state.current_phase = 3

        # Add pending effect
        self.state.pending_effects.append("DUMMY_EFFECT")

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, dm_ai_module.ActionType.RESOLVE_EFFECT)

if __name__ == '__main__':
    unittest.main()
