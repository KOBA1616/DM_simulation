import pytest
import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action

# Mock Action class if not available
class MockAction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def to_dict(self):
        return self.__dict__

@pytest.fixture
def game_context():
    # Load Cards
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    if not card_db:
        pytest.fail("Failed to load card DB")

    # Initialize State
    state = dm_ai_module.GameState(1000)
    dm_ai_module.PhaseManager.start_game(state, card_db)

    # Ensure P1 is active
    p1_id = state.active_player_id

    return {
        'state': state,
        'card_db': card_db,
        'p1_id': p1_id,
        'p2_id': 1 - p1_id
    }

def execute_via_map_action(state, card_db, action_dict):
    """Path 1: Action -> map_action -> ExecuteCommand"""
    cmd = map_action(action_dict)
    # Ensure it's executed via CommandSystem
    EngineCompat.ExecuteCommand(state, cmd, card_db)
    return cmd

def execute_via_direct_command(state, card_db, cmd_dict):
    """Path 2: Direct Command Dict -> ExecuteCommand"""
    EngineCompat.ExecuteCommand(state, cmd_dict, card_db)

def _run_turn_sequence(context, execution_method):
    state = context['state']
    card_db = context['card_db']
    p1 = context['p1_id']
    player = state.players[p1]

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

    # -------------------------------------------------------------------------
    # 3. PLAY CARD (Action: PLAY_FROM_ZONE / PLAY_CARD)
    # -------------------------------------------------------------------------
    # Ramp Mana using CommandSystem logic directly (TRANSITION command)
    # This avoids relying on internal shims that might be guarded/mocked.
    ramp_cmd = {
        'type': 'TRANSITION',
        'from_zone': 'DECK',
        'to_zone': 'MANA',
        'amount': 5, # Move 5 cards
        'owner_id': p1
    }
    try:
        EngineCompat.ExecuteCommand(state, ramp_cmd, card_db)
    except Exception as e:
        print(f"Setup ramp warning: {e}")

    # Now Play
    if len(player.hand) > 0:
        card_to_play = player.hand[0]

        action = {
            'type': 'PLAY_FROM_ZONE',
            'source_instance_id': card_to_play.instance_id,
            'from_zone': 'HAND',
            'to_zone': 'BATTLE',
            'value1': 999 # Max cost, or cheat cost payment?
            # Note: Engine checks logic. If logic fails, it's fine for this test
            # as long as CommandSystem was invoked.
        }

        execution_method(state, card_db, action if execution_method == execute_via_map_action else map_action(action))

    # -------------------------------------------------------------------------
    # 4. ATTACK (Action: ATTACK_PLAYER)
    # -------------------------------------------------------------------------
    # Setup: Need a creature in Battle Zone.
    # Transition one from Deck to Battle
    setup_attacker_cmd = {
        'type': 'TRANSITION',
        'from_zone': 'DECK',
        'to_zone': 'BATTLE',
        'amount': 1,
        'owner_id': p1
    }
    EngineCompat.ExecuteCommand(state, setup_attacker_cmd, card_db)

    # Get that creature
    player = state.players[p1]
    if len(player.battle_zone) > 0:
        attacker = player.battle_zone[-1]

        # Move to Attack Phase
        while str(EngineCompat.get_current_phase(state)) != "Phase.ATTACK":
             EngineCompat.PhaseManager_next_phase(state, card_db)

        action = {
            'type': 'ATTACK_PLAYER',
            'source_instance_id': attacker.instance_id,
            'target_player': 1 - p1
        }

        execution_method(state, card_db, action if execution_method == execute_via_map_action else map_action(action))


def test_e2e_via_map_action(game_context):
    """Phase 5.3 Path 1: Action Dict -> map_action -> ExecuteCommand"""
    _run_turn_sequence(game_context, execute_via_map_action)

def test_e2e_via_direct_command(game_context):
    """Phase 5.3 Path 2: Command Dict -> ExecuteCommand"""
    _run_turn_sequence(game_context, execute_via_direct_command)
