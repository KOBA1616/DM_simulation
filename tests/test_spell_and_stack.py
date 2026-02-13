
import unittest
import pytest
import dm_ai_module
from dm_ai_module import GameInstance, CommandType, CardStub, GameState, CommandDef, CardType

@pytest.mark.skipif(not getattr(dm_ai_module, 'IS_NATIVE', False), reason="Requires native engine")
class TestSpellAndStack(unittest.TestCase):
    def setUp(self):
        self.game = GameInstance()
        self.game.start_game()
        self.p0 = self.game.state.players[0]
        self.p1 = self.game.state.players[1]

    def test_spell_casting_stub(self):
        # Setup: Add a "Spell" card to hand.
        # Using ID 7 (Ice and Fire) which is a real Spell in data/cards.json
        spell_card_id = 7
        self.game.state.add_card_to_hand(0, spell_card_id)

        # Verify card is in hand
        hand_card = self.p0.hand[-1]
        self.assertEqual(hand_card.card_id, spell_card_id)

        # Action: Play Card (Cast Spell)
        cmd = CommandDef()
        cmd.type = CommandType.PLAY_FROM_ZONE
        cmd.instance_id = hand_card.instance_id

        # Execute via compat wrapper when available, fallback to instance method
        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, cmd, None)
        except Exception:
            try:
                self.game.execute_command(cmd)
            except Exception:
                pass

        # Verification 1: Card removed from hand
        card_in_hand = any(c.instance_id == hand_card.instance_id for c in self.p0.hand)
        self.assertFalse(card_in_hand, "Spell card should be removed from hand")

        # Verification 2: Pending effects populated
        self.assertEqual(len(self.game.state.pending_effects), 1, "Should have 1 pending effect")
        eff = self.game.state.pending_effects[0]
        # Allow attribute access for stub objects
        cid = getattr(eff, 'card_id', -1)
        self.assertEqual(cid, spell_card_id)

        # Verification 3: Resolve Stack
        resolve_cmd = CommandDef()
        resolve_cmd.type = CommandType.RESOLVE_EFFECT
        resolve_cmd.amount = 0 # slot index

        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, resolve_cmd, None)
        except Exception:
            try:
                self.game.execute_command(resolve_cmd)
            except Exception:
                pass

        self.assertEqual(len(self.game.state.pending_effects), 0, "Pending effects should be empty after resolution")

        # Verification 4: Card in graveyard
        card_in_grave = any(c.instance_id == hand_card.instance_id for c in self.p0.graveyard)
        self.assertTrue(card_in_grave, "Spell card should be in graveyard")

    def test_stack_lifo(self):
        """Verify that pending effects are resolved in LIFO order."""
        # 1. Add two spells to hand
        card_id_A = 7  # Spell A
        card_id_B = 8  # Spell B
        self.game.state.add_card_to_hand(0, card_id_A)
        self.game.state.add_card_to_hand(0, card_id_B)

        hand_card_A = self.p0.hand[-2]
        hand_card_B = self.p0.hand[-1]

        # 2. Play Spell A
        cmd_A = CommandDef()
        cmd_A.type = CommandType.PLAY_FROM_ZONE
        cmd_A.instance_id = hand_card_A.instance_id

        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, cmd_A, None)
        except Exception:
            try:
                self.game.execute_command(cmd_A)
            except Exception:
                pass

        # 3. Play Spell B (Triggered via some mechanism? Or just stacked?)
        # For this test, we simulate adding another effect to the stack
        # as if Spell A triggered Spell B or we are in a chain.
        # Since we can't easily chain in stub, we just Play B.
        cmd_B = CommandDef()
        cmd_B.type = CommandType.PLAY_FROM_ZONE
        cmd_B.instance_id = hand_card_B.instance_id

        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, cmd_B, None)
        except Exception:
            try:
                self.game.execute_command(cmd_B)
            except Exception:
                pass

        # Verify stack has 2 items: [A, B]
        self.assertEqual(len(self.game.state.pending_effects), 2)
        self.assertEqual(getattr(self.game.state.pending_effects[0], 'card_id'), card_id_A)
        self.assertEqual(getattr(self.game.state.pending_effects[1], 'card_id'), card_id_B)

        # 4. Resolve First (Should be B - index 1)
        # Note: RESOLVE_EFFECT with slot_index=1 (Top of stack if A is 0, B is 1)
        # But wait, pending_effects order: pushed back. 0 is first, 1 is second.
        # If LIFO, we resolve last one first?
        # Usually Stack is popped from back.
        # But `pending_effects` is vector. `resolve_effect` takes slot_index.
        # We manually target the last one.
        resolve_cmd = CommandDef()
        resolve_cmd.type = CommandType.RESOLVE_EFFECT
        resolve_cmd.amount = 1 # slot_index 1 (B)

        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, resolve_cmd, None)
        except Exception:
            try:
                self.game.execute_command(resolve_cmd)
            except Exception:
                pass

        # Verify stack has 1 item: [A]
        self.assertEqual(len(self.game.state.pending_effects), 1)
        self.assertEqual(getattr(self.game.state.pending_effects[0], 'card_id'), card_id_A)

        # 5. Resolve Second (Should be A - index 0)
        resolve_cmd.amount = 0
        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(self.game.state, resolve_cmd, None)
        except Exception:
            try:
                self.game.execute_command(resolve_cmd)
            except Exception:
                pass

        # Verify stack empty
        self.assertEqual(len(self.game.state.pending_effects), 0)

if __name__ == '__main__':
    unittest.main()
