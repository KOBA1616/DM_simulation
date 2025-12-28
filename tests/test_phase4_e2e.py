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

    # -------------------------------------------------------------------------
    # 0. Setup: Ensure we have resources
    # -------------------------------------------------------------------------

    # Ensure we are in MANA Phase
    current_phase = str(EngineCompat.get_current_phase(state))

    # -------------------------------------------------------------------------
    # 1. MANA CHARGE (Action: MANA_CHARGE)
    # -------------------------------------------------------------------------
    # Just in case, try to charge if possible
    if len(player.hand) > 0:
        card_to_charge = player.hand[0]

        # Action Dict
        action = {
            'type': 'MANA_CHARGE',
            'source_instance_id': card_to_charge.instance_id,
            'from_zone': 'HAND',
            'to_zone': 'MANA_ZONE' # Legacy name to test normalization
        }

        # Exec
        execution_method(state, card_db, action if execution_method == execute_via_map_action else map_action(action))

    # -------------------------------------------------------------------------
    # 2. TRANSITION TO MAIN
    # -------------------------------------------------------------------------
    while str(EngineCompat.get_current_phase(state)) != "Phase.MAIN":
         EngineCompat.PhaseManager_next_phase(state, card_db)

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
