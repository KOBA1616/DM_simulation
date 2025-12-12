
import pytest
import dm_ai_module
from dm_ai_module import (
    CardDefinition, CardData, GameState, Player, CostType, ReductionType,
    CostDef, CostReductionDef, FilterDef, CostPaymentSystem, Civilization, Zone
)

class TestCostPayment:
    def test_cost_definitions_structure(self):
        """Test that CostDef and CostReductionDef can be created and stored on CardDefinition."""

        # Create a CostDef
        cost_def = CostDef()
        cost_def.type = CostType.TAP_CARD
        cost_def.amount = 1

        f = FilterDef()
        f.zones = ["BATTLE_ZONE"]
        f.types = ["CREATURE"]
        cost_def.filter = f

        # Create a CostReductionDef
        reduction = CostReductionDef()
        reduction.type = ReductionType.ACTIVE_PAYMENT
        reduction.reduction_amount = 2
        reduction.min_mana_cost = 0
        reduction.unit_cost = cost_def
        reduction.name = "Hyper Energy"

        # Verify fields
        assert reduction.type == ReductionType.ACTIVE_PAYMENT
        assert reduction.reduction_amount == 2
        assert reduction.unit_cost.type == CostType.TAP_CARD

        # Check binding to CardDefinition
        card_def = CardDefinition()
        card_def.cost_reductions = [reduction]
        assert len(card_def.cost_reductions) == 1
        assert card_def.cost_reductions[0].name == "Hyper Energy"

    def test_calculate_max_units(self):
        """Test calculating available units (e.g. tappable creatures)"""
        state = GameState(100)
        state.setup_test_duel()

        # Setup: Player 0 has 3 creatures in Battle Zone
        # 2 are untapped, 1 is tapped
        pid = 0

        # Dummy Card DB
        card_db = {}

        # Create a dummy creature definition
        creature_def = CardDefinition()
        creature_def.id = 100
        creature_def.name = "Dummy Creature"
        creature_def.type = dm_ai_module.CardType.CREATURE
        creature_def.civilizations = [Civilization.FIRE]
        card_db[100] = creature_def

        # Add instances
        # Untapped
        state.add_test_card_to_battle(pid, 100, 0, False, False)
        state.add_test_card_to_battle(pid, 100, 1, False, False)
        # Tapped
        state.add_test_card_to_battle(pid, 100, 2, True, False)

        # Define reduction: Tap 1 Creature
        cost_def = CostDef()
        cost_def.type = CostType.TAP_CARD
        cost_def.amount = 1
        f = FilterDef()
        f.zones = ["BATTLE_ZONE"]
        # In test setup, simple check
        cost_def.filter = f

        reduction = CostReductionDef()
        reduction.type = ReductionType.ACTIVE_PAYMENT
        reduction.unit_cost = cost_def

        # Execute
        units = CostPaymentSystem.calculate_max_units(state, pid, reduction, card_db)

        # Expect 2 untapped creatures
        assert units == 2

    def test_can_pay_cost_hyper_energy(self):
        """Test payment logic for 'Scaling Hyper Energy' scenario"""
        state = GameState(100)
        state.setup_test_duel()
        pid = 0

        # Dummy DB
        card_db = {}

        # Creature Definition (for tap targets)
        creature_def = CardDefinition()
        creature_def.id = 100
        creature_def.civilizations = [Civilization.FIRE]
        card_db[100] = creature_def

        # Hyper Energy Creature Definition
        hyper_creature = CardDefinition()
        hyper_creature.id = 999
        hyper_creature.name = "Scaling Hyper Creature"
        hyper_creature.cost = 8
        hyper_creature.civilizations = [Civilization.FIRE]

        # Define Hyper Energy: -2 cost per tap, min 0
        cost_def = CostDef()
        cost_def.type = CostType.TAP_CARD
        cost_def.amount = 1
        cost_def.filter = FilterDef(zones=["BATTLE_ZONE"]) # Simplified filter

        reduction = CostReductionDef()
        reduction.type = ReductionType.ACTIVE_PAYMENT
        reduction.reduction_amount = 2
        reduction.min_mana_cost = 0
        reduction.unit_cost = cost_def

        hyper_creature.cost_reductions = [reduction]
        card_db[999] = hyper_creature

        # Scenario 1: No Mana, No Creatures -> Fail (Cost 8)
        assert not CostPaymentSystem.can_pay_cost(state, pid, hyper_creature, card_db)

        # Scenario 2: 8 Mana, No Creatures -> Success (Pay 8)
        state.clear_zone(pid, Zone.MANA)
        for i in range(8):
             state.add_card_to_mana(pid, 100, 1000+i)
             # state.players[pid].mana_zone[i].is_tapped = False # Defaults to false

        assert CostPaymentSystem.can_pay_cost(state, pid, hyper_creature, card_db)

        # Reset Mana
        state.clear_zone(pid, Zone.MANA)

        # Scenario 3: 4 Creatures (Untapped), 0 Mana -> Success (4 * 2 = 8 reduction)
        state.clear_zone(pid, Zone.BATTLE)
        for i in range(4):
            state.add_test_card_to_battle(pid, 100, 2000+i, False, False)

        assert CostPaymentSystem.can_pay_cost(state, pid, hyper_creature, card_db)

        # Scenario 4: 3 Creatures (Untapped), 2 Mana -> Success (3*2=6 red, 2 mana pay = 8 total)
        state.clear_zone(pid, Zone.BATTLE)
        for i in range(3):
            state.add_test_card_to_battle(pid, 100, 3000+i, False, False)

        state.clear_zone(pid, Zone.MANA)
        for i in range(2):
            state.add_card_to_mana(pid, 100, 4000+i) # ID 100 is Fire, matches civ

        assert CostPaymentSystem.can_pay_cost(state, pid, hyper_creature, card_db)

        # Scenario 5: 3 Creatures (Untapped), 1 Mana -> Fail (3*2=6 red, need 2 mana, have 1)
        # We need to be careful about accessing by reference vs copy.
        # Ideally use clear_zone and re-add 1 mana.
        state.clear_zone(pid, Zone.MANA)
        state.add_card_to_mana(pid, 100, 5000)

        assert not CostPaymentSystem.can_pay_cost(state, pid, hyper_creature, card_db)
