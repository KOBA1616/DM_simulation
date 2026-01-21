
import sys
import os
import unittest
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import dm_ai_module
except ImportError:
    print("dm_ai_module not found")
    sys.exit(1)

class TestSpellAndStack(unittest.TestCase):
    def setUp(self):
        self.game = dm_ai_module.GameInstance()
        self.game.start_game()
        self.state = self.game.state
        self.player_id = 0
        self.player = self.state.players[self.player_id]

    def test_spell_casting_flow(self):
        """
        Verify Issue 1 & 2: Spell Casting and Stack Processing.
        1. Add Spell (ID 2) to hand.
        2. Play Card -> Should go to pending_effects (Stack).
        3. Resolve Effect -> Should go to Graveyard.
        """
        # 1. Add Spell (ID 2) to hand
        # Note: In our stub, we hardcoded ID 2 as Spell in GameInstance.execute_action
        spell_card_id = 2

        # Manually add a specific card instance to hand for tracking
        instance_id = 999
        self.state.add_card_to_hand(self.player_id, spell_card_id, instance_id)

        # Verify it's in hand
        hand_card = None
        for c in self.player.hand:
            if c.instance_id == instance_id:
                hand_card = c
                break
        self.assertIsNotNone(hand_card, "Spell card should be in hand")
        initial_hand_size = len(self.player.hand)

        # 2. Play Card (Cast Spell)
        play_cmd = dm_ai_module.GameCommand()
        play_cmd.type = dm_ai_module.ActionType.PLAY_CARD
        play_cmd.source_instance_id = instance_id

        print(f"\n[Test] Executing PLAY_CARD for instance {instance_id} (ID {spell_card_id})")
        self.game.execute_action(play_cmd)

        # Verify removed from hand
        self.assertEqual(len(self.player.hand), initial_hand_size - 1, "Hand size should decrease by 1")

        # Verify added to pending_effects (Stack)
        self.assertEqual(len(self.state.pending_effects), 1, "Pending effects should have 1 card")
        self.assertEqual(self.state.pending_effects[0].instance_id, instance_id, "The pending effect should be our spell")

        print("[Test] Spell is now in pending_effects (Stack).")

        # 3. Resolve Effect
        resolve_cmd = dm_ai_module.GameCommand()
        resolve_cmd.type = dm_ai_module.ActionType.RESOLVE_EFFECT

        print("[Test] Executing RESOLVE_EFFECT")
        self.game.execute_action(resolve_cmd)

        # Verify removed from pending_effects
        self.assertEqual(len(self.state.pending_effects), 0, "Pending effects should be empty after resolution")

        # Verify added to Graveyard
        self.assertTrue(len(self.player.graveyard) > 0, "Graveyard should not be empty")
        # Check if our card is in graveyard
        in_grave = False
        for c in self.player.graveyard:
            if c.instance_id == instance_id:
                in_grave = True
                break
        self.assertTrue(in_grave, "Spell card should be in graveyard")

        print("[Test] Spell resolved and moved to Graveyard.")

if __name__ == '__main__':
    unittest.main()
