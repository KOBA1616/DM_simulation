
import pytest
import dm_ai_module

class TestLethalSolverAdvanced:
    @pytest.fixture
    def setup_game(self):
        # Initialize GameState with enough capacity
        state = dm_ai_module.GameState(100)
        state.setup_test_duel()
        state.active_player_id = 0
        state.turn_number = 1
        return state

    @pytest.fixture
    def card_db(self):
        db = {}

        # ID 1: Vanilla Attacker
        k1 = dm_ai_module.CardKeywords()
        db[1] = dm_ai_module.CardDefinition(1, "Attacker", "FIRE", [], 1, 1000, k1, [])

        # ID 2: Double Breaker
        k2 = dm_ai_module.CardKeywords()
        k2.double_breaker = True
        db[2] = dm_ai_module.CardDefinition(2, "Double Breaker", "FIRE", [], 6, 6000, k2, [])

        # ID 3: Unblockable
        k3 = dm_ai_module.CardKeywords()
        k3.unblockable = True
        db[3] = dm_ai_module.CardDefinition(3, "Unblockable", "WATER", [], 3, 3000, k3, [])

        # ID 4: Blocker
        k4 = dm_ai_module.CardKeywords()
        k4.blocker = True
        db[4] = dm_ai_module.CardDefinition(4, "Blocker", "LIGHT", [], 2, 2000, k4, [])

        return db

    def test_double_breaker_lethal(self, setup_game, card_db):
        state = setup_game
        # Scenario: 2 Shields. 1 DB, 1 Vanilla.
        # DB breaks 2. Vanilla hits player. Lethal.
        state.clear_zone(1, dm_ai_module.Zone.SHIELD)
        state.add_test_card_to_shield(1, 0, 0)
        state.add_test_card_to_shield(1, 0, 1)
        state.add_test_card_to_battle(0, 2, 100, False, False) # DB
        state.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla

        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == True

    def test_blocker_optimal_play(self, setup_game, card_db):
        state = setup_game
        # Scenario: 2 Shields, 1 Blocker.
        # Player has 1 DB, 2 Vanilla.
        # Optimal Defender blocks DB.
        # Remaining V, V break 2 shields. No direct attack.
        # Not Lethal.

        state.clear_zone(1, dm_ai_module.Zone.SHIELD)
        state.add_test_card_to_shield(1, 0, 0)
        state.add_test_card_to_shield(1, 0, 1)
        state.add_test_card_to_battle(1, 4, 200, False, False) # Blocker

        state.add_test_card_to_battle(0, 2, 100, False, False) # DB
        state.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla
        state.add_test_card_to_battle(0, 1, 102, False, False) # Vanilla

        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == False

    def test_blocker_overwhelmed(self, setup_game, card_db):
        state = setup_game
        # Scenario: 1 Shield, 1 Blocker.
        # Player has 1 DB, 2 Vanilla.
        # Optimal Defender blocks DB.
        # Remaining V, V.
        # V1 breaks 1 shield. V2 direct attack.
        # Lethal.

        state.clear_zone(1, dm_ai_module.Zone.SHIELD)
        state.add_test_card_to_shield(1, 0, 0)
        state.add_test_card_to_battle(1, 4, 200, False, False) # Blocker

        state.add_test_card_to_battle(0, 2, 100, False, False) # DB
        state.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla
        state.add_test_card_to_battle(0, 1, 102, False, False) # Vanilla

        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == True

    def test_unblockable_lethal(self, setup_game, card_db):
        state = setup_game
        # Scenario: 1 Shield, 1 Blocker.
        # Player has 1 Unblockable, 1 Vanilla.
        # Defender blocks Vanilla.
        # Unblockable breaks shield. No direct attack. Not Lethal.

        state.clear_zone(1, dm_ai_module.Zone.SHIELD)
        state.add_test_card_to_shield(1, 0, 0)
        state.add_test_card_to_battle(1, 4, 200, False, False) # Blocker

        state.add_test_card_to_battle(0, 3, 100, False, False) # Unblockable
        state.add_test_card_to_battle(0, 1, 101, False, False) # Vanilla

        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == False

        # Now add another Vanilla.
        # 1 Unblockable, 2 Vanilla vs 1 Blocker, 1 Shield.
        # Defender blocks V1.
        # V2 blocked? No, blocker used on V1.
        # Wait, if Defender blocks V1.
        # Unblockable breaks shield.
        # V2 direct attack.
        # Lethal.

        state.add_test_card_to_battle(0, 1, 102, False, False) # Vanilla
        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == True
