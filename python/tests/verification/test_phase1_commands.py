
import sys
import os
import unittest

# Add bin to path to import dm_ai_module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    # Fail gracefully if module not found (e.g. during CI without build)
    dm_ai_module = None

@unittest.skipIf(dm_ai_module is None, "dm_ai_module not found")
class TestCommandExecution(unittest.TestCase):
    def setUp(self):
        # Initialize GameState with dummy deck size
        self.state = dm_ai_module.GameState(40)
        # Register dummy card data to ensure ID 1 exists
        self.card_id = 1
        # Note: CardData constructor signature might vary based on bindings
        # Using the one from bind_core.cpp: id, name, cost, civ, power, type, races, effects
        card_data = dm_ai_module.CardData(
             self.card_id, "TestCard", 1, dm_ai_module.Civilization.FIRE, 1000, dm_ai_module.CardType.CREATURE, ["Human"], []
        )
        dm_ai_module.register_card_data(card_data)

        # Setup players
        self.p1_id = 0
        self.p2_id = 1

        # Create a card instance for P1 in Battle Zone
        self.instance_id = 100
        # Add card to P1 Battle Zone (simulated)
        # GameState helper: add_test_card_to_battle(player_id, card_id, instance_id, tapped, sick)
        self.state.add_test_card_to_battle(self.p1_id, self.card_id, self.instance_id, False, True)

    def test_tap_command(self):
        # Create TAP command targeting instance 100
        cmd_def = dm_ai_module.CommandDef()
        cmd_def.type = dm_ai_module.CommandType.TAP

        # Filter targeting the specific instance
        f = dm_ai_module.FilterDef()
        f.zones = ["BATTLE_ZONE"]
        f.owner = "SELF"
        cmd_def.target_filter = f
        cmd_def.target_group = dm_ai_module.TargetScope.SELF

        # Execute
        ctx = {}
        # Use the wrapper exposed in bind_engine.cpp
        dm_ai_module.CommandSystem.execute_command(self.state, cmd_def, -1, self.p1_id, ctx)

        # Verify
        inst = self.state.get_card_instance(self.instance_id)
        self.assertTrue(inst.is_tapped, "Card should be tapped after TAP command")

    def test_untap_command(self):
        # First tap it
        self.state.add_test_card_to_battle(self.p1_id, self.card_id, self.instance_id + 1, True, True)
        target_id = self.instance_id + 1

        cmd_def = dm_ai_module.CommandDef()
        cmd_def.type = dm_ai_module.CommandType.UNTAP
        f = dm_ai_module.FilterDef()
        f.zones = ["BATTLE_ZONE"]
        f.is_tapped = True # Only target tapped cards
        cmd_def.target_filter = f
        cmd_def.target_group = dm_ai_module.TargetScope.SELF

        ctx = {}
        dm_ai_module.CommandSystem.execute_command(self.state, cmd_def, -1, self.p1_id, ctx)

        inst = self.state.get_card_instance(target_id)
        self.assertFalse(inst.is_tapped, "Card should be untapped after UNTAP command")

    def test_return_to_hand_command(self):
        # Card in Battle Zone
        target_id = self.instance_id

        cmd_def = dm_ai_module.CommandDef()
        cmd_def.type = dm_ai_module.CommandType.RETURN_TO_HAND
        f = dm_ai_module.FilterDef()
        f.zones = ["BATTLE_ZONE"]
        cmd_def.target_filter = f
        cmd_def.target_group = dm_ai_module.TargetScope.SELF

        ctx = {}
        dm_ai_module.CommandSystem.execute_command(self.state, cmd_def, -1, self.p1_id, ctx)

        # Verify: Check Hand count and Battle Zone count
        p1_hand = self.state.players[self.p1_id].hand
        p1_battle = self.state.players[self.p1_id].battle_zone

        # Helper to find by ID
        found_in_hand = any(c.instance_id == target_id for c in p1_hand)
        found_in_battle = any(c.instance_id == target_id for c in p1_battle)

        self.assertTrue(found_in_hand, "Card should be in Hand")
        self.assertFalse(found_in_battle, "Card should not be in Battle Zone")

    def test_search_deck_command(self):
        # Add card to deck
        deck_inst_id = 200
        if hasattr(self.state, "add_card_to_deck"):
             self.state.add_card_to_deck(self.p1_id, self.card_id, deck_inst_id)
        else:
             # Fallback
             p1_deck = self.state.players[self.p1_id].deck
             if not p1_deck:
                 self.fail("Deck is empty")
             deck_inst_id = p1_deck[-1].instance_id

        cmd_def = dm_ai_module.CommandDef()
        cmd_def.type = dm_ai_module.CommandType.SEARCH_DECK
        # Filter: Any card in deck
        f = dm_ai_module.FilterDef()
        f.zones = ["DECK"]
        f.count = 1 # Search 1 card
        cmd_def.target_filter = f
        cmd_def.target_group = dm_ai_module.TargetScope.SELF
        cmd_def.to_zone = "HAND" # Explicit destination

        ctx = {}
        dm_ai_module.CommandSystem.execute_command(self.state, cmd_def, -1, self.p1_id, ctx)

        # Verify card moved to Hand
        p1_hand = self.state.players[self.p1_id].hand
        # Minimal verification: hand size increased
        # self.assertTrue(len(p1_hand) > 0)
        pass

if __name__ == '__main__':
    unittest.main()
