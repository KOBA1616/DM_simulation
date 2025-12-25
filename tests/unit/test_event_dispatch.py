
import sys
import os

# Add build artifacts to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

import dm_ai_module
import pytest

class TestEventDispatch:
    def test_zone_enter_trigger(self):
        """
        Verify that moving a card to Battle Zone triggers ON_PLAY effects.
        We use a card that has CIP ability.
        """
        try:
            # Load definitions
            dm_ai_module.JsonLoader.load_cards("data/cards.json")

            # Register dummy card BEFORE creating GameInstance
            card_id = 9999

            # Create effects list
            effect_def = dm_ai_module.EffectDef()
            effect_def.trigger = dm_ai_module.TriggerType.ON_PLAY
            action_def = dm_ai_module.ActionDef()
            action_def.type = dm_ai_module.EffectActionType.DRAW_CARD
            effect_def.actions = [action_def]

            cdata = dm_ai_module.CardData(
                card_id,
                "Test CIP Creature",
                3,
                "WATER",
                2000,
                "CREATURE",
                ["Cyber Lord"],
                [effect_def]
            )

            dm_ai_module.register_card_data(cdata)

            # Use GameInstance(seed) which uses Singleton CardRegistry
            game = dm_ai_module.GameInstance(1)
            state = game.state

            # Now add this card to hand
            player_id = 0
            instance_id = 0
            state.add_card_to_hand(player_id, card_id, instance_id)

            # Execute Move (Transition) to Battle Zone
            cmd = dm_ai_module.TransitionCommand(
                instance_id,
                dm_ai_module.Zone.HAND,
                dm_ai_module.Zone.BATTLE,
                player_id,
                -1
            )
            state.execute_command(cmd)

            # Check Pending Effects
            pending_info = state.get_pending_effects_info()
            print(f"Pending Effects: {pending_info}")

            found_trigger = False
            for p in pending_info:
                print(f"Pending Type: {p['type']}")
                if p['source_instance_id'] == instance_id:
                     found_trigger = True

            assert found_trigger, "Trigger was not queued upon entering battle zone!"

        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
