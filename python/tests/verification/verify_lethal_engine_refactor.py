
import os
import sys
import unittest

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.types import CardDB, GameState
from dm_toolkit.ai.agent.network import AlphaZeroNetwork

class TestLethalSolverRefactor(unittest.TestCase):
    def setUp(self):
        # Create Dummy DB
        self.card_db = {}

        # 1. Vanilla Creature (Fire)
        c1 = dm_ai_module.CardDefinition()
        c1.name = "Vanilla"
        c1.cost = 1
        c1.power = 1000
        c1.civilizations = [dm_ai_module.Civilization.FIRE]
        c1.type = dm_ai_module.CardType.CREATURE
        self.card_db[1] = c1

        # 2. Speed Attacker (Fire)
        c2 = dm_ai_module.CardDefinition()
        c2.name = "Speedy"
        c2.cost = 1
        c2.power = 1000
        c2.civilizations = [dm_ai_module.Civilization.FIRE]
        c2.type = dm_ai_module.CardType.CREATURE
        c2.keywords.speed_attacker = True
        self.card_db[2] = c2

        # 3. Evolution (Fire)
        c3 = dm_ai_module.CardDefinition()
        c3.name = "Evo"
        c3.cost = 1
        c3.power = 5000
        c3.civilizations = [dm_ai_module.Civilization.FIRE]
        c3.type = dm_ai_module.CardType.EVOLUTION_CREATURE
        c3.keywords.evolution = True
        c3.keywords.double_breaker = True
        c3.races = ["Human"]
        self.card_db[3] = c3

        # 4. Blocker (Light)
        c4 = dm_ai_module.CardDefinition()
        c4.name = "Blocker"
        c4.cost = 1
        c4.power = 2000
        c4.civilizations = [dm_ai_module.Civilization.LIGHT]
        c4.type = dm_ai_module.CardType.CREATURE
        c4.keywords.blocker = True
        self.card_db[4] = c4

        # 5. Base for Evo (Fire, Human)
        c5 = dm_ai_module.CardDefinition()
        c5.name = "Base"
        c5.cost = 1
        c5.power = 1000
        c5.civilizations = [dm_ai_module.Civilization.FIRE]
        c5.type = dm_ai_module.CardType.CREATURE
        c5.races = ["Human"]
        self.card_db[5] = c5

        self.solver = dm_ai_module.LethalSolver

    def test_speed_attacker_from_hand(self):
        """Test that solver sees lethal via playing a Speed Attacker from hand."""
        state = dm_ai_module.GameState(100)
        dm_ai_module.initialize_card_stats(state, self.card_db, 100)

        # Setup: P1 has Speedy in hand, Opponent has 0 shields, 0 blockers.
        # Need to be in Main Phase.
        state.active_player_id = 0
        state.current_phase = dm_ai_module.Phase.MAIN

        # Add SA to hand
        state.add_card_to_hand(0, 2, 0) # player, card_id, instance_id
        # Add Mana to pay (ID 1 is Fire)
        state.add_card_to_mana(0, 1, 1)

        # Ensure opponent has 0 shields
        self.assertEqual(len(state.players[1].shield_zone), 0)

        is_lethal = self.solver.is_lethal(state, self.card_db)
        self.assertTrue(is_lethal, "Should find lethal by playing Speed Attacker")

    def test_evolution_rush(self):
        """Test that solver sees lethal via Evolution on existing creature."""
        state = dm_ai_module.GameState(100)
        dm_ai_module.initialize_card_stats(state, self.card_db, 100)

        state.active_player_id = 0
        state.current_phase = dm_ai_module.Phase.MAIN

        # Base in Battle Zone (summoning sick or not, doesn't matter for Evo, but let's say sick)
        state.add_test_card_to_battle(0, 5, 0, False, True) # id 5 (Human), sick=True

        # Evo in Hand
        state.add_card_to_hand(0, 3, 1) # id 3 (Evo Human)

        # Mana (Fire)
        state.add_card_to_mana(0, 1, 2)

        self.assertEqual(len(state.players[1].shield_zone), 0)

        is_lethal = self.solver.is_lethal(state, self.card_db)
        self.assertTrue(is_lethal, "Should find lethal by Evolving")

    def test_blocked_not_lethal(self):
        """Test that solver correctly identifies non-lethal if opponent can block."""
        state = dm_ai_module.GameState(100)
        dm_ai_module.initialize_card_stats(state, self.card_db, 100)

        state.active_player_id = 0
        state.current_phase = dm_ai_module.Phase.ATTACK # Start in Attack to simplify

        # Attacker (Power 1000)
        state.add_test_card_to_battle(0, 1, 0, False, False)

        # Blocker (Power 2000, Untapped)
        state.add_test_card_to_battle(1, 4, 1, False, False)

        # Opponent has 0 shields (so if attack goes through, we win)
        # But Blocker prevents it.

        is_lethal = self.solver.is_lethal(state, self.card_db)
        self.assertFalse(is_lethal, "Should NOT be lethal because of blocker")

    def test_complex_shield_break(self):
        """Test breaking shields to win."""
        state = dm_ai_module.GameState(100)
        dm_ai_module.initialize_card_stats(state, self.card_db, 100)

        state.active_player_id = 0
        state.current_phase = dm_ai_module.Phase.ATTACK

        # 2 Attackers
        state.add_test_card_to_battle(0, 1, 0, False, False)
        state.add_test_card_to_battle(0, 1, 1, False, False)

        # Opponent has 1 shield
        state.add_card_to_shield(1, 1, 2)

        is_lethal = self.solver.is_lethal(state, self.card_db)
        self.assertTrue(is_lethal, "Should be lethal (Break shield + Direct Attack)")

if __name__ == '__main__':
    unittest.main()
