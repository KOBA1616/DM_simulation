
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

        # 4: Double Breaker
        kw_db = dm_ai_module.CardKeywords()
        kw_db.double_breaker = True
        self.card_db[4] = dm_ai_module.CardDefinition(
            4, "Breaker", "FIRE", ["Dragon"], 5, 6000,
            kw_db, []
        )
        self.card_db[4].type = dm_ai_module.CardType.CREATURE

        self.game = dm_ai_module.GameState(100)
        self.game.setup_test_duel()

    def test_simple_lethal(self):
        self.game.clear_shield_zone(1)
        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 1, 100, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal)

    def test_not_lethal_due_to_shields(self):
        self.game.clear_shield_zone(1)
        self.game.add_test_card_to_shield(1, 1, 200) # 1 Shield
        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 1, 100, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal)
        self.game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal)

    def test_blocker_stops_lethal(self):
        self.game.clear_shield_zone(1)
        self.game.add_test_card_to_battle(1, 3, 300, False, True) # Blocker
        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 1, 100, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal)
        self.game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal)

    def test_double_breaker_logic(self):
        """
        1 Double Breaker vs 2 Shields -> Not Lethal (0 left for direct).
        1 Double Breaker + 1 Vanilla vs 2 Shields -> Lethal.
        """
        self.game.clear_shield_zone(1)
        self.game.add_test_card_to_shield(1, 1, 200)
        self.game.add_test_card_to_shield(1, 1, 201) # 2 Shields

        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 4, 100, False, False) # Double Breaker

        # DB breaks 2 shields. Remaining attackers 0. Direct attack impossible.
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal, "Double Breaker alone against 2 shields is NOT lethal")

        # Add Vanilla
        self.game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertTrue(is_lethal, "DB + Vanilla against 2 shields IS lethal")

    def test_smart_blocking_logic(self):
        """
        1 Double Breaker + 1 Vanilla vs 2 Shields + 1 Blocker.
        Opponent should block DB. Vanilla breaks 1 shield. 1 shield left. Not Lethal.
        If logic was dumb (blocks vanilla), DB breaks 2 shields -> Lethal.
        """
        self.game.clear_shield_zone(1)
        self.game.add_test_card_to_shield(1, 1, 200)
        self.game.add_test_card_to_shield(1, 1, 201) # 2 Shields
        self.game.add_test_card_to_battle(1, 3, 300, False, True) # Blocker

        self.game.turn_number = 2
        self.game.active_player_id = 0
        self.game.add_test_card_to_battle(0, 4, 100, False, False) # DB
        self.game.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        self.assertFalse(is_lethal, "Blocker should block DB to prevent lethal")

if __name__ == "__main__":
    unittest.main()
