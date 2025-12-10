
import pytest
import dm_ai_module

class TestLethalSolverOrdering:
    @pytest.fixture
    def setup_game(self):
        state = dm_ai_module.GameState(100)
        state.setup_test_duel()
        state.active_player_id = 0
        state.turn_number = 1
        return state

    @pytest.fixture
    def card_db(self):
        db = {}

        # ID 1: Blockable Double Breaker
        k1 = dm_ai_module.CardKeywords()
        k1.double_breaker = True
        db[1] = dm_ai_module.CardDefinition(1, "Double Breaker", "FIRE", [], 6, 6000, k1, [])

        # ID 2: Unblockable Single Breaker
        k2 = dm_ai_module.CardKeywords()
        k2.unblockable = True
        db[2] = dm_ai_module.CardDefinition(2, "Unblockable", "WATER", [], 3, 3000, k2, [])

        return db

    def test_unblockable_breaker_ordering(self, setup_game, card_db):
        state = setup_game
        # Scenario: 2 Shields.
        # Player has:
        # 1. Unblockable Single Breaker (ID 2)
        # 2. Blockable Double Breaker (ID 1)
        # Opponent has NO Blockers.

        # Logic with Incorrect Sorting:
        # Unblockable (Single) added first. Blockable (Double) added second.
        # Loop:
        #   Unblockable hits -> Breaks 1 shield. Remaining Shields = 1.
        #   Double hits -> Breaks 1 shield. Remaining Shields = 0.
        #   Result: No Direct Attack. False.

        # Logic with Correct Sorting (Desc Breaker):
        # Successful list: [Double(2), Unblockable(1)].
        # Loop:
        #   Double hits -> Breaks 2 shields. Remaining Shields = 0.
        #   Unblockable hits -> Direct Attack.
        #   Result: True.

        state.clear_zone(1, dm_ai_module.Zone.SHIELD)
        state.add_test_card_to_shield(1, 0, 0)
        state.add_test_card_to_shield(1, 0, 1)

        state.add_test_card_to_battle(0, 1, 100, False, False) # DB (Blockable)
        state.add_test_card_to_battle(0, 2, 101, False, False) # Unblockable (Single)

        # Ensure no blockers
        state.clear_zone(1, dm_ai_module.Zone.BATTLE)

        assert dm_ai_module.LethalSolver.is_lethal(state, card_db) == True
