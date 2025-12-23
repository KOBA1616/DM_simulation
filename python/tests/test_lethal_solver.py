
import sys
import os
import unittest
import pytest

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
# Add build directory to path (where bindings usually are)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

class TestLethalSolver(unittest.TestCase):
    def setUp(self):
        self.card_db = {}
        # 1: Vanilla Attacker
        self.card_db[1] = dm_ai_module.CardDefinition(
            1, "Vanilla", "FIRE", ["Human"], 2, 2000,
            dm_ai_module.CardKeywords(), []
        )
        self.card_db[1].type = dm_ai_module.CardType.CREATURE

        # 2: Speed Attacker
        kw_sa = dm_ai_module.CardKeywords()
        kw_sa.speed_attacker = True
        self.card_db[2] = dm_ai_module.CardDefinition(
            2, "Speedy", "FIRE", ["Human"], 3, 3000,
            kw_sa, []
        )
        self.card_db[2].type = dm_ai_module.CardType.CREATURE

        # 3: Blocker
        kw_blocker = dm_ai_module.CardKeywords()
        kw_blocker.blocker = True
        self.card_db[3] = dm_ai_module.CardDefinition(
            3, "Blocky", "LIGHT", ["Guardian"], 2, 4000,
            kw_blocker, []
        )
        self.card_db[3].type = dm_ai_module.CardType.CREATURE

        self.game = dm_ai_module.GameState(100)
        self.game.setup_test_duel()

    def test_simple_lethal(self):
        """
        Opponent has 0 shields.
        I have 1 attacker (can attack).
        Lethal = True.
        """
        self.game.clear_zone(1, dm_ai_module.Zone.SHIELD) # Opponent no shields

        # Add Attacker for me (Turn 1 played, current turn 2 -> Summoning Sickness gone)
        # Assuming turn_number 2
        self.game.turn_number = 2
        self.game.active_player_id = 0

        # Add card to battle: pid=0, cid=1, iid=100, tapped=False, sick=False
        self.game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal with 1 attacker vs 0 shields")

    def test_not_lethal_due_to_shields(self):
        """
        Opponent has 1 shield.
        I have 1 attacker.
        Need 2 attackers (1 for shield, 1 for direct).
        Lethal = False.
        """
        self.game.clear_zone(1, dm_ai_module.Zone.SHIELD)
        self.game.add_test_card_to_shield(1, 1, 200) # 1 Shield

        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal, "Should NOT be lethal (1 attacker vs 1 shield)")

        # Add another attacker
        self.game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal (2 attackers vs 1 shield)")

    def test_blocker_stops_lethal(self):
        """
        Opponent has 0 shields, 1 blocker.
        I have 1 attacker.
        Lethal = False (Blocker blocks).
        """
        self.game.clear_zone(1, dm_ai_module.Zone.SHIELD)

        # Add Opponent Blocker
        self.game.add_test_card_to_battle(1, 3, 300, False, True) # Blocker (sick, but can block)

        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal, "Blocker should prevent lethal")

        # Add another attacker
        self.game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        # 2 Attackers vs 1 Blocker + 0 Shields.
        # 1 blocked, 1 direct attack. Lethal = True.
        self.assertTrue(is_lethal, "2 Attackers vs 1 Blocker should be lethal")

    def test_speed_attacker_counts(self):
        """
        Attacker has Summoning Sickness but is Speed Attacker.
        """
        self.game.clear_zone(1, dm_ai_module.Zone.SHIELD)

        # Add SA (sick=True)
        self.game.add_test_card_to_battle(0, 2, 100, False, True)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal, "Speed Attacker should count despite sickness")

    def test_summoning_sickness_prevents_lethal(self):
        """
        Attacker has Summoning Sickness (and not SA).
        """
        self.game.clear_zone(1, dm_ai_module.Zone.SHIELD)

        # Add Vanilla (sick=True)
        self.game.add_test_card_to_battle(0, 1, 100, False, True)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal, "Sick creature should not attack")

if __name__ == "__main__":
    unittest.main()
