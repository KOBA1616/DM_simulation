import pytest

from dm_toolkit.gui.game_session import GameSession


def test_gamesession_api_surface_exists():
    gs = GameSession()

    # Public callbacks
    assert hasattr(gs, 'callback_update_ui')
    assert hasattr(gs, 'callback_log')
    assert hasattr(gs, 'callback_input_request')
    assert hasattr(gs, 'callback_action_executed')

    # Controller and engine interaction helpers
    assert hasattr(gs, 'controller')
    assert hasattr(gs, 'initialize_game')
    assert hasattr(gs, 'reset_game')
    assert hasattr(gs, 'step_game')
    assert hasattr(gs, 'execute_command')

    # Async worker hooks
    assert hasattr(gs, '_start_ai_worker')
    assert hasattr(gs, '_stop_ai_worker')
    assert hasattr(gs, '_ai_thread')
