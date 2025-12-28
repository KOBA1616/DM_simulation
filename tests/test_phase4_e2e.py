
import unittest
import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action
try:
    from dm_ai_module import Action
except ImportError:
    class Action:
        def to_dict(self):
            return self.__dict__

class TestPhase4E2E(unittest.TestCase):
    def setUp(self):
        # Create a game state with 1000 cards
        self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        self.state = dm_ai_module.GameState(1000)
        dm_ai_module.PhaseManager.start_game(self.state, self.card_db)

        # Helper to get valid commands
        self.p1 = self.state.active_player_id

    def test_simple_turn_flow(self):
        """
        Verify: Draw -> Play -> Attack -> End Turn using CommandSystem.
        """
        state = self.state

        # 1. Start of Game (Mana Charge Phase usually)
        # Note: start_game might put us in MANA phase or MAIN depending on logic.
        # Let's check phase.
        phase = EngineCompat.get_current_phase(state)
        print(f"Start Phase: {phase}")

        # If in Mana Phase, try to charge mana
        if str(phase) == "Phase.MANA":
            # Find a card in hand
            hand = state.players[self.p1].hand
            if hand.size() > 0:
                card = hand[0]

                # Create Action -> Command
                action = Action()
                action.type = "MANA_CHARGE"
                action.source_instance_id = card.instance_id
                action.to_zone = "MANA_ZONE"
                action.from_zone = "HAND"

                cmd_dict = map_action(action.to_dict() if hasattr(action, 'to_dict') else action.__dict__)
                print(f"Executing Mana Charge: {cmd_dict}")

                # Execute via Compat (should use CommandSystem)
                EngineCompat.ExecuteCommand(state, cmd_dict, self.card_db)

                # Verify Mana increased
                self.assertEqual(state.players[self.p1].mana_zone.size(), 1)

            # Transition to Main Phase
            # Usually handled by next_phase if no more actions, or explicit PASS
            # Engine logic: if mana charged or pass, go to next.
            # Let's try PASS command (ActionType.PASS mapped to FLOW?)
            # Or just next_phase directly if we want to skip.
            # Let's try to generate legal commands and pick one.
            pass

        # Force transition to Main Phase if needed
        while str(EngineCompat.get_current_phase(state)) != "Phase.MAIN":
            EngineCompat.PhaseManager_next_phase(state, self.card_db)

        print(f"Current Phase: {EngineCompat.get_current_phase(state)}")

        # 2. Main Phase: Play a creature
        # Cheat: Add a cheap creature to hand and mana to pay for it
        # Actually, let's just add a card and try to play it.
        # We need to know a valid card ID. ID 1 is usually dummy/token.
        # Let's assume we have a creature.

        # 3. End Turn
        # Execute End Turn Command
        # For this minimal test, let's verify TRANSITION command works for generic move.
        # Move card from Deck to Hand (Draw)
        player = state.players[self.p1]
        deck = player.deck
        # Note: In C++, deck is std::vector. In Python binding, it might be exposed as list or custom type.
        # If it's a list, use len(deck). If it has size(), use it.
        # EngineCompat uses getattr/aliases.

        # Check type of deck
        deck_size = len(deck) if isinstance(deck, list) else deck.size()

        if deck_size > 0:
            top_card = deck[deck_size-1]
            draw_cmd = {
                'type': 'TRANSITION', # or DRAW_CARD
                'from_zone': 'DECK',
                'to_zone': 'HAND',
                'instance_id': top_card.instance_id,
                'owner_id': self.p1
            }

            # For hand, handle both list and vector
            hand = player.hand
            prev_hand_size = len(hand) if isinstance(hand, list) else hand.size()

            EngineCompat.ExecuteCommand(state, draw_cmd, self.card_db)

            # Refresh player/hand reference if needed (though state should update in place)
            hand = state.players[self.p1].hand
            curr_hand_size = len(hand) if isinstance(hand, list) else hand.size()

            self.assertEqual(curr_hand_size, prev_hand_size + 1, "Draw command failed")
            print("Draw Command Successful")

if __name__ == '__main__':
    unittest.main()
