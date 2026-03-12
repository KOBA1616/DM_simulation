import time
from dm_toolkit.gui.game_session import GameSession


def test_game_session_ai_worker_calls_update_ui():
    calls = []

    def ui_callback():
        calls.append(time.time())

    gs = GameSession(callback_update_ui=ui_callback)

    # Initialize with native DB and instance; if native module missing, skip
    try:
        gs.initialize_game(seed=1)
    except RuntimeError:
        # dm_ai_module not available in this environment; mark as xfail
        import pytest

        pytest.skip("dm_ai_module not available")

    # Set both players to AI so worker will run
    try:
        gs.set_player_mode(0, 'AI')
        gs.set_player_mode(1, 'AI')
    except Exception:
        pass

    # Start auto-stepping which should spawn the AI worker
    gs.is_running = True
    gs._auto_step_loop()

    # Wait a short time to allow worker to call UI
    time.sleep(0.15)

    # Stop worker and assert UI was called at least once
    gs._stop_ai_worker()
    gs.is_running = False

    assert len(calls) >= 1
