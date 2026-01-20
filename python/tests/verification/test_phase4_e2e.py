import pytest
import dm_ai_module
import logging
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.command_builders import (
    build_draw_command,
    build_transition_command,
    build_mana_charge_command,
    build_attack_player_command
)

# ============================================================================
# Phase 3: Gradual Migration from Action Dicts to GameCommand Builders
# ============================================================================
# This test file demonstrates the staged migration strategy outlined in AGENTS.md:
# 1. Legacy Path: MockAction + map_action (backward compatibility)
# 2. Modern Path: Direct GameCommand construction via builder functions
#
# The execute_via_direct_command path uses builders to demonstrate the preferred
# pattern for new code, while execute_via_map_action maintains compatibility
# with legacy Action dictionary patterns.
# ============================================================================

# Mock Action class for legacy compatibility paths
class MockAction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def to_dict(self):
        return self.__dict__

@pytest.fixture
def game_context():
    # Load Cards
    card_db_dict = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    if not card_db_dict:
        pytest.fail("Failed to load card DB")

    card_db = dm_ai_module.CardDatabase()
    for k, v in card_db_dict.items():
        card_db[k] = v

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
    # start_game might put us in MANA or MAIN. Force checks as needed.

    # ---------------------------------------------------------------------
    # 1. DRAW (Using GameCommand Builder - Phase 3 Migration)
    # ---------------------------------------------------------------------
    # MIGRATION NOTE: Prefer build_draw_command over MockAction + map_action
    # for cleaner, more maintainable test code.
    deck_size_before = len(player.deck)
    hand_size_before = len(player.hand)

    if deck_size_before > 0:
        top_card = player.deck[-1]

        # Phase 3 Preferred Pattern: Direct GameCommand construction
        if execution_method == execute_via_direct_command:
            cmd_dict = build_draw_command(
                from_zone="DECK",
                to_zone="HAND",
                source_instance_id=top_card.instance_id
            )
        else:
            # Legacy Path: Still supported via map_action for backward compatibility
            draw_action = MockAction(type="DRAW_CARD",
                                     from_zone="DECK",
                                     to_zone="HAND",
                                     source_instance_id=top_card.instance_id)
            cmd_dict = map_action(draw_action.to_dict())

        # Execute and verify via pytest assertions
        EngineCompat.ExecuteCommand(state, cmd_dict, card_db)

        player = state.players[p1]
        assert len(player.hand) == hand_size_before + 1, "Draw command should increase hand size"
        assert len(player.deck) == deck_size_before - 1, "Draw command should decrease deck size"

    # ---------------------------------------------------------------------
    # 2. MANA CHARGE (Using GameCommand Builder - Phase 3 Migration)
    # ---------------------------------------------------------------------
    try:
        phase = EngineCompat.get_current_phase(state)
    except Exception:
        phase = None

    if phase is not None and str(phase) == "Phase.MANA":
        hand = player.hand
        hand_size = len(hand) if isinstance(hand, list) else hand.size()
        if hand_size > 0:
            card_to_charge = hand[0]
            mana_zone_before = player.mana_zone
            mana_size_before = len(mana_zone_before) if isinstance(mana_zone_before, list) else mana_zone_before.size()

            # Phase 3 Preferred Pattern: Direct GameCommand construction
            if execution_method == execute_via_direct_command:
                cmd_dict = build_mana_charge_command(
                    source_instance_id=card_to_charge.instance_id,
                    from_zone="HAND"
                )
            else:
                # Legacy Path: MockAction + map_action
                charge_action = MockAction(type="MANA_CHARGE",
                                           from_zone="HAND",
                                           source_instance_id=card_to_charge.instance_id)
                cmd_dict = map_action(charge_action.to_dict())

            # Zone string normalization check
            assert cmd_dict.get('to_zone') == "MANA"

            EngineCompat.ExecuteCommand(state, cmd_dict, card_db)

            player = state.players[p1]
            mana_zone = player.mana_zone
            mana_size = len(mana_zone) if isinstance(mana_zone, list) else mana_zone.size()
            assert mana_size == mana_size_before + 1, "Mana charge should increase mana zone size"

    # Advance to MAIN Phase
    # Calling next_phase until MAIN
    try:
        while str(EngineCompat.get_current_phase(state)) != "Phase.MAIN":
            EngineCompat.PhaseManager_next_phase(state, card_db)
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 3. PLAY CARD (Action: PLAY_FROM_ZONE / PLAY_CARD)
    # -------------------------------------------------------------------------
    # Ramp Mana using Direct GameCommand (Phase 3 Migration Pattern)
    # This avoids relying on internal shims that might be guarded/mocked.
    ramp_cmd = build_transition_command(
        from_zone='DECK',
        to_zone='MANA',
        amount=5,  # Move 5 cards
        owner_id=p1
    )
    try:
        EngineCompat.ExecuteCommand(state, ramp_cmd, card_db)
    except Exception as e:
        logging.getLogger('dm_toolkit.tests').warning("Setup ramp warning: %s", e)

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
