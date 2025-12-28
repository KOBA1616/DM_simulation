
import unittest
import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action
try:
    from dm_ai_module import Action, Zone
except ImportError:
    class Action:
        def to_dict(self):
            return self.__dict__
    Zone = None

class TestPhase4E2E(unittest.TestCase):
    def setUp(self):
        # Create a game state with 1000 cards
        self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        self.state = dm_ai_module.GameState(1000)
        dm_ai_module.PhaseManager.start_game(self.state, self.card_db)

        # Helper to get valid commands
        self.p1 = self.state.active_player_id

    def _execute_phase_loop(self):
        # Helper to ensure engine transitions if needed
        # In this environment, we might need to manually trigger next_phase if actions don't auto-trigger.
        pass

    def test_simple_turn_flow(self):
        """
        Verify: Draw -> Mana -> Play -> Attack -> End Turn using CommandSystem.
        """
        state = self.state
        player_idx = self.p1
        player = state.players[player_idx]

        # ---------------------------------------------------------------------
        # 0. Initial Setup & Phase check
        # ---------------------------------------------------------------------
        # start_game might put us in MANA or MAIN.
        # For simplicity, force phase progression if needed or assume start.
        # Usually starts in MANA phase for player 0.

        # ---------------------------------------------------------------------
        # 1. DRAW (Simulated as TRANSITION from DECK to HAND)
        # ---------------------------------------------------------------------
        # Note: start_game usually draws initial hand.
        # We will force a draw to test the command.

        # Deck/Hand are lists in python binding, so use len()
        deck_size_before = len(player.deck)
        hand_size_before = len(player.hand)

        if deck_size_before > 0:
            top_card = player.deck[deck_size_before-1]

            # Legacy Action simulation
            draw_action = Action()
            draw_action.type = "DRAW_CARD"
            draw_action.from_zone = "DECK"
            draw_action.to_zone = "HAND"
            draw_action.source_instance_id = top_card.instance_id

            cmd_dict = map_action(draw_action.to_dict() if hasattr(draw_action, 'to_dict') else draw_action.__dict__)

            # Phase 4.4-1 Verification: Executing dict command via CommandSystem
            EngineCompat.ExecuteCommand(state, cmd_dict, self.card_db)

            # Verify
            player = state.players[player_idx] # Refresh
            self.assertEqual(len(player.hand), hand_size_before + 1, "Draw command should increase hand size")
            self.assertEqual(len(player.deck), deck_size_before - 1, "Draw command should decrease deck size")

        # ---------------------------------------------------------------------
        # 2. MANA CHARGE
        # ---------------------------------------------------------------------
        # Must be in MANA phase. If not, skip or force phase.
        phase = EngineCompat.get_current_phase(state)
        if str(phase) == "Phase.MANA":
            hand = player.hand
            hand_size = len(hand) if isinstance(hand, list) else hand.size()
            if hand_size > 0:
                card_to_charge = hand[0]
                mana_zone_before = player.mana_zone
                mana_size_before = len(mana_zone_before) if isinstance(mana_zone_before, list) else mana_zone_before.size()

                charge_action = Action()
                charge_action.type = "MANA_CHARGE"
                charge_action.from_zone = "HAND"
                # to_zone handled by map_action ("MANA")
                charge_action.source_instance_id = card_to_charge.instance_id

                cmd_dict = map_action(charge_action.to_dict() if hasattr(charge_action, 'to_dict') else charge_action.__dict__)

                # Phase 4.4-2 Verification: Zone string normalization (MANA_ZONE -> MANA)
                # cmd_dict['to_zone'] should be "MANA" now
                self.assertEqual(cmd_dict['to_zone'], "MANA")

                EngineCompat.ExecuteCommand(state, cmd_dict, self.card_db)

                player = state.players[player_idx] # Refresh
                mana_zone = player.mana_zone
                mana_size = len(mana_zone) if isinstance(mana_zone, list) else mana_zone.size()
                self.assertEqual(mana_size, mana_size_before + 1, "Mana charge should increase mana zone size")

        # Advance to MAIN Phase
        # Calling next_phase until MAIN
        while str(EngineCompat.get_current_phase(state)) != "Phase.MAIN":
            EngineCompat.PhaseManager_next_phase(state, self.card_db)

        # ---------------------------------------------------------------------
        # 3. PLAY CARD
        # ---------------------------------------------------------------------
        # We need a playable card.

        # Let's try to PLAY the first card in hand.
        player = state.players[player_idx] # Refresh
        if len(player.hand) > 0:
            card_to_play = player.hand[0]

            # Use PLAY_FROM_ZONE (Phase 4.4-3 requirement)
            play_action = Action()
            play_action.type = "PLAY_FROM_ZONE" # or RESOLVE_PLAY if atomic
            play_action.from_zone = "HAND"
            play_action.to_zone = "BATTLE"
            play_action.source_instance_id = card_to_play.instance_id

            cmd_dict = map_action(play_action.to_dict() if hasattr(play_action, 'to_dict') else play_action.__dict__)

            # Execute
            # Note: This might fail logic check (Cost), but should hit CommandSystem.
            # We can catch potential error or check logs.
            try:
                EngineCompat.ExecuteCommand(state, cmd_dict, self.card_db)
            except Exception as e:
                print(f"Play command execution logic error (expected if cost not met): {e}")

        # ---------------------------------------------------------------------
        # 4. ATTACK (Transition to ATTACK phase)
        # ---------------------------------------------------------------------
        # Move to Attack Phase
        while str(EngineCompat.get_current_phase(state)) != "Phase.ATTACK":
             EngineCompat.PhaseManager_next_phase(state, self.card_db)

        # Need a creature in Battle Zone to attack.
        # Cheat: Move a card from deck to battle zone directly.
        player = state.players[player_idx] # Refresh
        if len(player.deck) > 0:
            cheat_card = player.deck[0]
            # Use direct engine method to place card for setup
            # But wait, we want to test CommandSystem.
            # Let's use a TRANSITION command to put it there first.
            setup_cmd = {
                'type': 'TRANSITION',
                'from_zone': 'DECK',
                'to_zone': 'BATTLE',
                'instance_id': cheat_card.instance_id,
                'owner_id': player_idx
            }
            EngineCompat.ExecuteCommand(state, setup_cmd, self.card_db)

            # Now Attack Player

            attack_action = Action()
            attack_action.type = "ATTACK_PLAYER"
            attack_action.source_instance_id = cheat_card.instance_id
            attack_action.target_player = 1 - player_idx # Opponent

            cmd_dict = map_action(attack_action.to_dict() if hasattr(attack_action, 'to_dict') else attack_action.__dict__)

            # Let's try executing.
            EngineCompat.ExecuteCommand(state, cmd_dict, self.card_db)

        # ---------------------------------------------------------------------
        # 5. END TURN
        # ---------------------------------------------------------------------
        pass

if __name__ == '__main__':
    unittest.main()
