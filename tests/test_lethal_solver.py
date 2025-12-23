
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

    def _make_game(self):
        game = dm_ai_module.GameState(100)
        # Empty state by default
        game.active_player_id = 0
        game.turn_number = 2
        game.current_phase = dm_ai_module.Phase.ATTACK
        # Initialize zones to empty just in case (though constructor does it)
        # We assume they are empty.
        return game

    def test_simple_lethal(self):
        """
        Opponent has 0 shields.
        I have 1 attacker (can attack).
        Lethal = True.
        """
        game = self._make_game()
        # No shields for P1 (default)

        # Add Attacker for P0
        game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal with 1 attacker vs 0 shields")

    def test_not_lethal_due_to_shields(self):
        """
        Opponent has 1 shield.
        I have 1 attacker.
        Need 2 attackers (1 for shield, 1 for direct).
        Lethal = False.
        """
        game = self._make_game()

        # Add Shield for P1
        game.add_card_to_shield(1, 1, 200)

        # Add Attacker P0
        game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertFalse(is_lethal, "Should NOT be lethal (1 attacker vs 1 shield)")

        # Add another attacker
        game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal (2 attackers vs 1 shield)")

    def test_blocker_stops_lethal(self):
        """
        Opponent has 0 shields, 1 blocker.
        I have 1 attacker.
        Lethal = False (Blocker blocks).
        """
        game = self._make_game()

        # Add Opponent Blocker
        game.add_test_card_to_battle(1, 3, 300, False, True) # Blocker (sick, but can block)

        # Add Attacker P0
        game.add_test_card_to_battle(0, 1, 100, False, False)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertFalse(is_lethal, "Blocker should prevent lethal")

        # Add another attacker
        game.add_test_card_to_battle(0, 1, 101, False, False)
        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertTrue(is_lethal, "2 Attackers vs 1 Blocker should be lethal")

    def test_speed_attacker_counts(self):
        """
        Attacker has Summoning Sickness but is Speed Attacker.
        """
        game = self._make_game()

        # Add SA (sick=True)
        game.add_test_card_to_battle(0, 2, 100, False, True)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertTrue(is_lethal, "Speed Attacker should count despite sickness")

    def test_summoning_sickness_prevents_lethal(self):
        """
        Attacker has Summoning Sickness (and not SA).
        """
        game = self._make_game()

        # Add Vanilla (sick=True)
        game.add_test_card_to_battle(0, 1, 100, False, True)

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertFalse(is_lethal, "Sick creature should not attack")

    def test_double_breaker_ordering(self):
        """
        Opponent has 1 shield, 1 Blocker (Power 7000).
        Attackers:
        A: Double Breaker (6000)
        B: Vanilla (2000)

        If I attack with DB first -> Blocked (DB dies). Vanilla hits shield. 0 Shields left. Not Lethal.
        If I attack with Vanilla first -> Blocked? (Vanilla dies). DB hits shield (Overkill). 0 Shields left. Not Lethal.
        If Opponent lets Vanilla hit -> 0 Shields. DB blocked.

        Actually, we need 3 attackers here?
        Let's try: 0 Shields. 1 Blocker.
        Attackers: DB, Vanilla.
        Attack DB -> Blocked. Vanilla -> Hits. Win.
        Attack Vanilla -> Blocked. DB -> Hits. Win.

        Wait, if 1 Shield.
        Attackers: DB (6000), Vanilla (2000), Vanilla2 (2000).
        Blocker (7000).

        Path 1: Attack DB. Blocked (DB dies). Vanilla1 hits shield (0 left). Vanilla2 hits player (Win).
        Path 2: Attack Vanilla1.
            Option A (Block): Vanilla1 dies. DB hits shield. Vanilla2 hits player. Win.
            Option B (No Block): Vanilla1 hits shield. DB blocked. Vanilla2 hits player. Win.

        So this scenario is Lethal regardless of order.
        """
        game = self._make_game()
        game.add_card_to_shield(1, 1, 200) # 1 Shield
        game.add_test_card_to_battle(1, 3, 300, False, True) # Blocker 4000
        # Wait, Blocker 3 is 4000 power.
        # DB is 6000. DB kills Blocker!

        # Let's make Blocker stronger
        self.card_db[5] = dm_ai_module.CardDefinition(
            5, "BigBlocker", "LIGHT", ["Guardian"], 5, 7000,
            self.card_db[3].keywords, []
        )
        # Update game blocker to 7000
        game = self._make_game()
        game.add_card_to_shield(1, 1, 200)
        game.add_test_card_to_battle(1, 5, 300, False, True) # 7000 Blocker

        game.add_test_card_to_battle(0, 4, 100, False, False) # DB 6000
        game.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla 2000
        game.add_test_card_to_battle(0, 1, 102, False, False) # Vanilla 2000

        is_lethal = dm_ai_module.LethalSolver.is_lethal(game, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal with sufficient attackers")

if __name__ == "__main__":
    unittest.main()
