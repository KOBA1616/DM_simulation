import unittest
import dm_ai_module
from dm_ai_module import GameInstance, CommandType

class TestCommandPath(unittest.TestCase):
    def test_apply_move_mana_charge(self):
        # Setup
        gi = GameInstance()
        gi.state.setup_test_duel()
        gi.state.set_deck(0, [1, 2, 3, 4, 5] * 8)
        gi.state.set_deck(1, [1, 2, 3, 4, 5] * 8)

        # Manually set phase to MANA
        gi.state.current_phase = 2 # MANA
        gi.state.active_player_id = 0

        # Add card to hand
        card = gi.state.add_card_to_hand(0, 100, 2000) # player, card_id, instance_id

        # Create Command dict
        cmd = {
            "type": CommandType.MANA_CHARGE,
            "instance_id": 2000,
            "amount": 0
        }

        # Execute via apply_move
        if hasattr(gi.state, 'apply_move'):
            print("Testing apply_move...")
            gi.state.apply_move(cmd)

            # Verify: Card moved to Mana
            mana = gi.state.get_zone(0, 2) # MANA
            # Fallback get_zone might return list of objects or list of IDs
            # Native returns list of IDs. Fallback returns list of CardStub.

            found = False
            for c in mana:
                iid = getattr(c, 'instance_id', c)
                if iid == 2000:
                    found = True
                    break

            if not found:
                self.fail(f"Card 2000 not found in mana zone. Content: {mana}")
            else:
                print("Card successfully moved to Mana Zone.")

            # Verify Enums in fallback
            try:
                t = CommandType.ATTACK_PLAYER
                print(f"CommandType.ATTACK_PLAYER exists: {t}")
            except AttributeError:
                self.fail("CommandType.ATTACK_PLAYER missing in dm_ai_module fallback")

        else:
            self.fail("apply_move not available on GameState")

if __name__ == '__main__':
    unittest.main()
