
import unittest
import sys
import os

# Ensure bin is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin')))

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found. Please build the project first.")
    sys.exit(1)

from dm_ai_module import (
    GameState,
    CardData,
    EffectDef,
    ActionDef,
    FilterDef,
    EffectActionType,
    TriggerType,
    Civilization,
    GenericCardSystem,
    JsonLoader,
    CommandType,
    TargetScope,
    CardRegistry
)

class TestTriggerMigration(unittest.TestCase):
    def setUp(self):
        # Initialize GameState with enough space
        self.state = GameState(1000)
        self.card_id = 999

        # Use valid constructor
        self.card_def = CardData(
            self.card_id,
            "Test Card",
            1,
            [Civilization.FIRE],
            1000,
            "CREATURE",
            [],
            [], # effects
            []  # reaction_abilities
        )

        # Register card data
        dm_ai_module.register_card_data(self.card_def)

        # Add a card instance to hand to act as source
        self.state.add_card_to_hand(0, self.card_id, 0)

        # Helper map for execution context
        self.ctx = {}

        # Fix for Argument Type Error:
        # Instead of creating a manual dict which pybind might reject if types don't match exactly,
        # we retrieve the official registry map which we just updated.
        self.card_db = CardRegistry.get_all_definitions()

    def test_draw_card_migration(self):
        """Verify that EffectActionType.DRAW_CARD executes correctly via CommandSystem."""
        # Setup Deck
        self.state.add_card_to_deck(0, self.card_id, 1)
        self.state.add_card_to_deck(0, self.card_id, 2)

        initial_hand = len(self.state.players[0].hand)
        initial_deck = len(self.state.players[0].deck)

        # Create Action: Draw 2
        action = ActionDef()
        action.type = EffectActionType.DRAW_CARD
        action.value1 = 2

        # Execute
        GenericCardSystem.resolve_action_with_db(self.state, action, 0, self.card_db, self.ctx)

        # Verify
        self.assertEqual(len(self.state.players[0].hand), initial_hand + 2)
        self.assertEqual(len(self.state.players[0].deck), initial_deck - 2)

    def test_mana_charge_migration(self):
        """Verify that EffectActionType.ADD_MANA executes correctly via CommandSystem."""
        # Setup Deck
        self.state.add_card_to_deck(0, self.card_id, 10)

        initial_mana = len(self.state.players[0].mana_zone)

        # Create Action: Add Mana 1
        action = ActionDef()
        action.type = EffectActionType.ADD_MANA
        action.value1 = 1

        GenericCardSystem.resolve_action_with_db(self.state, action, 0, self.card_db, self.ctx)

        self.assertEqual(len(self.state.players[0].mana_zone), initial_mana + 1)

    def test_destroy_card_migration(self):
        """Verify that EffectActionType.DESTROY executes correctly via CommandSystem."""
        # Setup Battle Zone
        self.state.add_test_card_to_battle(0, self.card_id, 100, False, False)

        initial_grave = len(self.state.players[0].graveyard)

        # Create Action: Destroy Creature
        action = ActionDef()
        action.type = EffectActionType.DESTROY
        action.scope = TargetScope.SELF
        # Filter targeting the creature
        f = FilterDef()
        f.zones = ["BATTLE_ZONE"]

        action.filter = f

        # We simulate this being triggered by instance 100
        GenericCardSystem.resolve_action_with_db(self.state, action, 100, self.card_db, self.ctx)

        # Check graveyard
        self.assertEqual(len(self.state.players[0].graveyard), initial_grave + 1)
        # Verify instance 100 is in grave

        in_grave = False
        for c in self.state.players[0].graveyard:
            if c.instance_id == 100:
                in_grave = True
                break
        self.assertTrue(in_grave, "Card 100 should be in graveyard")

if __name__ == '__main__':
    unittest.main()
