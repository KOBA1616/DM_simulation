import pytest
from dm_ai_module import GameState, MutateCommand, MutationType, Zone
from dm_ai_module import Action, ActionType

class TestCommandSystem:
    def setup_method(self):
        self.state = GameState(40)
        # Initialize deck with some dummy cards
        for i in range(40):
            self.state.add_card_to_deck(0, 1, i) # player, card_id, instance_id
            self.state.add_card_to_deck(1, 1, 40+i)

    def test_draw_command(self):
        pass

    def test_mutate_tap_command(self):
         # Setup a card in battle zone
         self.state.add_test_card_to_battle(0, 1, 100, False, False)

         cmd = MutateCommand(100, MutationType.TAP, 0, "")
         # execute is on GameState
         self.state.execute_command(cmd)

         inst = self.state.get_card_instance(100)
         assert inst.is_tapped
