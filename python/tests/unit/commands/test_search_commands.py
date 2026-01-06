import pytest
from dm_ai_module import GameState, ShuffleCommand, SearchDeckCommand, Zone, CommandSystem, TransitionCommand

class TestSearchCommands:
    def setup_method(self):
        self.state = GameState(40)
        # Initialize deck with 10 cards (ID 1-10)
        for i in range(10):
            self.state.add_card_to_deck(0, 1, i)

    def test_shuffle_command(self):
        # Capture order before
        deck_before = [c.instance_id for c in self.state.players[0].deck]

        cmd = ShuffleCommand(0)
        self.state.execute_command(cmd)

        deck_after = [c.instance_id for c in self.state.players[0].deck]

        # It's possible shuffle results in same order, but unlikely for 10 cards
        # We just verify same elements exist
        assert set(deck_before) == set(deck_after)
        assert len(deck_before) == len(deck_after)

    def test_search_deck_command(self):
        # Target specific cards to move to Hand
        targets = [0, 1] # Instance IDs

        cmd = SearchDeckCommand(0, targets, Zone.HAND)
        self.state.execute_command(cmd)

        # Verify cards moved to Hand
        hand = [c.instance_id for c in self.state.players[0].hand]
        assert 0 in hand
        assert 1 in hand

        # Verify removed from Deck
        deck = [c.instance_id for c in self.state.players[0].deck]
        assert 0 not in deck
        assert 1 not in deck

        # Verify shuffle happened (implied if logic works, but hard to test randomness here without seed control)
        # We just ensure operation completed safely.
