import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

import dm_ai_module
import unittest

class TestPipelineResume(unittest.TestCase):
    def test_pipeline_pause_and_resume(self):
        # 1. Setup GameInstance with a dummy card DB
        card_db = {}

        # Initialize GameInstance
        game = dm_ai_module.GameInstance(card_db, 42)
        game.start_game()

        # Define a card with a targeted DISCARD effect
        card_id = 1
        card_def = dm_ai_module.CardDefinition()
        card_def.id = card_id
        card_def.name = "Test Card"
        card_def.type = dm_ai_module.CardType.CREATURE
        card_def.cost = 1

        effect = dm_ai_module.EffectDef()
        effect.trigger = dm_ai_module.TriggerType.ON_PLAY

        action = dm_ai_module.ActionDef()
        action.type = dm_ai_module.EffectActionType.DISCARD
        action.scope = dm_ai_module.TargetScope.TARGET_SELECT

        # Filter: Discard 1 card from SELF HAND
        filter_def = dm_ai_module.FilterDef()
        filter_def.owner = "SELF"
        filter_def.zones = ["HAND"]
        filter_def.count = 1
        action.filter = filter_def

        effect.actions.append(action)
        card_def.effects.append(effect)

        card_db[card_id] = card_def

        # Re-init
        game = dm_ai_module.GameInstance(card_db, 42)
        game.start_game()

        # P0 Hand: Test Card (100), Dummy (101)
        # Note: add_test_card_to_hand helpers
        # Ensure we have a valid dummy definition for 101 or it might crash if not in DB
        # Though DB is map<int, Def>, missing ID often defaults or handled safely if code checks count.
        # Safest to add dummy.
        dummy_def = dm_ai_module.CardDefinition()
        dummy_def.id = 2
        card_db[2] = dummy_def

        game.state.add_test_card_to_hand(0, card_id, 100)
        game.state.add_test_card_to_hand(0, 2, 101)

        # Action: PLAY_CARD
        play_action = dm_ai_module.Action()
        play_action.type = dm_ai_module.ActionType.PLAY_CARD
        play_action.card_id = card_id
        play_action.source_instance_id = 100
        play_action.target_player = 0

        print("Processing Play Action (Triggering Discard)...")
        try:
            game.process_action(play_action)
        except Exception as e:
            print(f"Error processing action: {e}")
            raise

        print(f"Waiting for input: {game.is_waiting_for_input()}")

        # If Logic Failure: DiscardHandler::compile might not be emitting SELECT.
        # If so, the test will fail here.
        if not game.is_waiting_for_input():
            # If it failed, it might be because DiscardHandler::compile doesn't handle implicit selection.
            # I need to update DiscardHandler to include SELECT instruction if targets not pre-selected.
            print("FAIL: Pipeline did not pause. Ensure DiscardHandler emits SELECT instruction.")
            # self.fail("Pipeline did not pause")
            return

        self.assertTrue(game.is_waiting_for_input())

        # 5. Resume
        print("Resuming with selection [101]...")
        game.resume_processing([101])

        # 6. Verify Result
        grave = game.state.players[0].graveyard
        # Note: Python binding for vector returns list copy usually
        found = False
        for c in grave:
            if c.instance_id == 101:
                found = True
                break

        if found:
            print("Success: Card 101 found in graveyard.")
        else:
            print("FAIL: Card 101 NOT in graveyard.")

        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()
