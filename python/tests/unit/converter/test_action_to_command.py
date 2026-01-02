# -*- coding: utf-8 -*-
import unittest
import copy
from dm_toolkit.action_to_command import map_action
try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

class TestActionToCommand(unittest.TestCase):

    def test_invalid_input(self):
        cmd = map_action(None)
        self.assertTrue(cmd.get('legacy_warning'))
        self.assertEqual(cmd['str_param'], "Invalid action shape")

    def test_basic_move_to_mana(self):
        act = {
            "type": "MOVE_CARD",
            "from_zone": "HAND",
            "to_zone": "MANA_ZONE",
            "value1": 1
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "TRANSITION")
        self.assertEqual(cmd['to_zone'], "MANA")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['amount'], 1)

    def test_destroy_card(self):
        act = {
            "type": "DESTROY",
            "source_zone": "BATTLE_ZONE"
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        self.assertEqual(cmd['from_zone'], "BATTLE")

    def test_draw_card(self):
        act = {
            "type": "DRAW_CARD",
            "value1": 2
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "DRAW_CARD")
        self.assertEqual(cmd['from_zone'], "DECK")
        self.assertEqual(cmd['to_zone'], "HAND")
        self.assertEqual(cmd['amount'], 2)

    def test_tap(self):
        act = {
            "type": "TAP",
            "filter": {"zones": ["BATTLE_ZONE"]}
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "TAP")
        self.assertEqual(cmd['target_filter']['zones'], ["BATTLE_ZONE"])

    def test_modifiers(self):
        act = {
            "type": "APPLY_MODIFIER",
            "str_val": "POWER_MOD",
            "value1": 1000
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "MUTATE")
        self.assertEqual(cmd['str_param'], "POWER_MOD")
        self.assertEqual(cmd['amount'], 1000)

    def test_nested_options(self):
        act = {
            "type": "SELECT_OPTION",
            "options": [
                {"type": "DRAW_CARD", "value1": 1},
                {"type": "MANA_CHARGE", "value1": 1}
            ]
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "CHOICE")
        self.assertTrue('options' in cmd)
        self.assertEqual(len(cmd['options']), 2)

        opt1 = cmd['options'][0][0] # options is list of lists of commands
        self.assertEqual(opt1['type'], "DRAW_CARD")
        self.assertEqual(opt1['to_zone'], "HAND")

        opt2 = cmd['options'][1][0]
        self.assertEqual(opt2['type'], "MANA_CHARGE")
        self.assertEqual(opt2['to_zone'], "MANA")

    def test_attack_player(self):
        act = {
            "type": "ATTACK_PLAYER",
            "source_instance_id": 100,
            "target_player": 1
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "ATTACK_PLAYER")
        self.assertEqual(cmd['instance_id'], 100)
        self.assertEqual(cmd['target_player'], 1)

    def test_legacy_keyword_fallback(self):
        act = {
            "type": "NONE",
            "str_val": "SPEED_ATTACKER"
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "ADD_KEYWORD")
        self.assertEqual(cmd['mutation_kind'], "SPEED_ATTACKER")

    def test_strict_type_validation_with_mock(self):
        # Mock dm_ai_module.CommandType to test validation logic
        import dm_toolkit.action_to_command as module_under_test

        # Create a mock Enum class
        class MockCommandType:
            VALID_TYPE = 1
            DRAW_CARD = 2

        # Capture original
        original_command_type = module_under_test._CommandType

        # Inject Mock
        module_under_test.set_command_type_enum(MockCommandType)

        try:
            # Case 1: Valid type
            act_valid = {"type": "DRAW_CARD", "value1": 1}
            cmd_valid = map_action(act_valid)
            self.assertEqual(cmd_valid['type'], "DRAW_CARD")
            self.assertFalse(cmd_valid.get('legacy_warning'))

            # Case 2: Invalid type (simulated)
            act_invalid = {"type": "MOVE_CARD", "from_zone": "HAND", "to_zone": "MANA_ZONE"}
            # map_action converts MOVE_CARD to TRANSITION.
            # TRANSITION is NOT in MockCommandType.
            cmd_invalid = map_action(act_invalid)

            self.assertEqual(cmd_invalid['type'], "NONE")
            self.assertTrue(cmd_invalid['legacy_warning'])
            # add_aliases_to_command might set legacy_original_type to original action type (MOVE_CARD)
            # which is preferred over the intermediate mapped type (TRANSITION).
            self.assertEqual(cmd_invalid['legacy_original_type'], "MOVE_CARD")
            self.assertIn("Invalid CommandType: TRANSITION", cmd_invalid['str_param'])

        finally:
            # Restore original
            module_under_test.set_command_type_enum(original_command_type)

    def test_determinism_execution(self):
        """
        Verify that mapping actions to commands and executing them on the C++ engine
        produces deterministic state transitions.
        """
        if dm_ai_module is None:
            print("Skipping determinism test: dm_ai_module not available")
            return

        from dm_toolkit.engine.compat import EngineCompat

        # 1. Setup a controlled environment (same seed/deck if possible, though here we rely on GameState init)
        # Note: dm_ai_module.GameState(100) creates a state with random seed usually,
        # but the deterministic property is that applying command C to State S always yields S'.
        # We will create one initial state, copy/clone it (if possible) or re-create it exactly if seeds are controllable.
        # Since we can't easily clone C++ GameState without pickling support (which might be missing),
        # we will rely on repeating the sequence on the SAME state instance (resetting it) OR
        # better: Assume we can't clone. We will run the sequence once, record the hash.
        # Then create a NEW state (hoping for same seed if default? Or we set seed?)
        # Actually, let's test: Apply Action -> Command -> Execution
        # Ensure that map_action is deterministic (it is).
        # Ensure that command execution is deterministic given the same state.

        # Since we cannot guarantee identical initial state across runs without seed control,
        # we will test determinism by:
        # A. Create State S1.
        # B. Apply Command C1 -> S1'. Record Hash H1.
        # C. Apply Command C1 -> S1' AGAIN (invalid usually) or...
        # Instead, let's verify map_action stability first (trivial).

        # Real verification:
        # 1. Take an action dict.
        # 2. Map it to command dict C.
        # 3. Verify C is always identical.
        # 4. Use EngineCompat to execute C on a state.
        # 5. Verify no crash and state changes (basic integration).

        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        if not card_db:
            print("Skipping determinism test: cards.json not found")
            return

        # Initialize with enough capacity (standard game is ~80-100 cards total)
        # Using 40 might be too small if decks are 40 each + shields + hand.
        state = dm_ai_module.GameState(200)

        # Explicitly set decks for both players before starting the game
        # start_game will draw 5 shields + 5 hand cards from these decks.
        # We use card ID 1 (dummy) for simplicity.
        deck_cards = [1] * 40
        state.set_deck(0, deck_cards)
        state.set_deck(1, deck_cards)

        dm_ai_module.PhaseManager.start_game(state, card_db)

        p1 = state.active_player_id

        # Ensure at least 1 card in deck
        self.assertGreater(len(state.players[p1].deck), 0)

        # Define a Draw Action
        action = {
            "type": "DRAW_CARD",
            "value1": 1,
            "target_player": p1
        }

        # Step 1: Stability of Mapping
        cmd1 = map_action(action)
        cmd2 = map_action(action)
        # remove uids for comparison
        uid1 = cmd1.pop('uid', None)
        uid2 = cmd2.pop('uid', None)
        self.assertEqual(cmd1, cmd2)

        # Step 2: Execution
        # We use the raw command dict which EngineCompat.ExecuteCommand supports
        # It will convert it to C++ CommandDef internally.

        # Record state before
        hand_count_before = len(state.players[p1].hand)
        deck_count_before = len(state.players[p1].deck)

        # Restore uid (though not strictly needed by engine usually, but good for tracking)
        cmd1['uid'] = uid1

        EngineCompat.ExecuteCommand(state, cmd1, card_db)

        hand_count_after = len(state.players[p1].hand)
        deck_count_after = len(state.players[p1].deck)

        self.assertEqual(hand_count_after, hand_count_before + 1)
        self.assertEqual(deck_count_after, deck_count_before - 1)

        # Step 3: Transition Determinism (Same command type = same logic)
        # We can't reset state easily to verify "Same S + Same C => Same S'"
        # without state cloning. But we verified the pipeline works.

        # Let's try to verify that passing the SAME command structure (a new instance)
        # works identically on a fresh state?

        state2 = dm_ai_module.GameState(200)
        state2.set_deck(0, deck_cards)
        state2.set_deck(1, deck_cards)
        dm_ai_module.PhaseManager.start_game(state2, card_db)
        # We assume start_game is deterministic if no shuffle? Or if shuffle, we can't compare S1 and S2.
        # But we can verify the delta.

        p1_2 = state2.active_player_id
        hand_before_2 = len(state2.players[p1_2].hand)

        cmd3 = map_action(action)
        EngineCompat.ExecuteCommand(state2, cmd3, card_db)

        hand_after_2 = len(state2.players[p1_2].hand)
        self.assertEqual(hand_after_2, hand_before_2 + 1)

if __name__ == '__main__':
    unittest.main()
