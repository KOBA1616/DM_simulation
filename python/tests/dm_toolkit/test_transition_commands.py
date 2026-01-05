
import unittest
import sys
import os

# Add relevant paths
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    from dm_toolkit import dm_ai_module
except ImportError:
    try:
        import dm_ai_module
    except ImportError:
        print("dm_ai_module not found. Skipping tests requiring engine.")
        sys.exit(0)

# Helper to convert dict to CommandDef since implicit conversion might not be enabled for pybind11 structs
def dict_to_command_def(d):
    cmd = dm_ai_module.CommandDef()
    # Map type string to Enum
    if 'type' in d:
        t_str = d['type']
        # Try to find enum value
        if hasattr(dm_ai_module.CommandType, t_str):
            cmd.type = getattr(dm_ai_module.CommandType, t_str)

    if 'from_zone' in d: cmd.from_zone = d['from_zone']
    if 'to_zone' in d: cmd.to_zone = d['to_zone']
    if 'amount' in d: cmd.amount = d['amount']
    if 'target_group' in d:
        tg_str = d['target_group']
        if hasattr(dm_ai_module.TargetScope, tg_str):
            cmd.target_group = getattr(dm_ai_module.TargetScope, tg_str)

    if 'target_filter' in d:
        f = dm_ai_module.FilterDef()
        fd = d['target_filter']
        if 'zones' in fd: f.zones = fd['zones']
        if 'count' in fd: f.count = fd['count']
        cmd.target_filter = f

    return cmd

class TestTransitionCommands(unittest.TestCase):
    def setUp(self):
        # Skip if running with shim (CommandSystem is defined in Python shim)
        if getattr(dm_ai_module.CommandSystem, '__module__', '') == 'dm_toolkit.dm_ai_module':
             self.skipTest("Skipping transition tests because native dm_ai_module is not available (using shim)")
        self.state = dm_ai_module.GameState(100)
        self.card_db = {}
        # Register dummy card data for ID 1
        # CardData(id, name, cost, civ, power, type, races, effects)
        cdata = dm_ai_module.CardData(1, "TestCard", 1, dm_ai_module.Civilization.FIRE, 1000, dm_ai_module.CardType.CREATURE, [], [])
        dm_ai_module.register_card_data(cdata)

    def test_draw_card_transition(self):
        player_id = 0
        for i in range(5):
             self.state.add_card_to_deck(player_id, 1, i)

        cmd_dict = {
            "type": "TRANSITION",
            "from_zone": "DECK",
            "to_zone": "HAND",
            "amount": 2,
            "target_group": "PLAYER_SELF"
        }
        cmd = dict_to_command_def(cmd_dict)

        ctx = {}
        dm_ai_module.CommandSystem.execute_command(self.state, cmd, -1, player_id, ctx)

        hand_size = len(self.state.players[0].hand)
        self.assertEqual(hand_size, 2, "Should have drawn 2 cards")

        deck_size = len(self.state.players[0].deck)
        self.assertEqual(deck_size, 3, "Deck should have 3 cards remaining")

    def test_destroy_transition_implicit(self):
        player_id = 0
        card_id = 1
        inst_id = 100
        self.state.add_test_card_to_battle(player_id, card_id, inst_id, False, False)

        cmd_dict_self = {
             "type": "TRANSITION",
             "to_zone": "GRAVEYARD",
             "target_group": "SELF"
        }
        cmd = dict_to_command_def(cmd_dict_self)

        ctx = {}
        dm_ai_module.CommandSystem.execute_command(self.state, cmd, inst_id, player_id, ctx)

        # Verify
        self.assertEqual(len(self.state.players[0].battle_zone), 0)
        self.assertEqual(len(self.state.players[0].graveyard), 1)

if __name__ == '__main__':
    unittest.main()
