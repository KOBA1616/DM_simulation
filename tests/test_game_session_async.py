import threading
import time

from dm_toolkit.gui.game_session import GameSession


class FakeState:
    def __init__(self):
        self.game_over = False

    def is_human_player(self, pid):
        return False


class FakeInstance:
    def __init__(self):
        self.state = FakeState()
        self._called = 0

    def step(self):
        # On first call, mark called and return True; then set game_over to stop worker
        self._called += 1
        if self._called >= 1:
            # ensure the worker sees game over after notifying UI
            self.state.game_over = True
        return True


def test_ai_worker_calls_update_ui_and_exits():
    called = threading.Event()

    def ui_callback():
        called.set()

    gs = GameSession(callback_update_ui=ui_callback)

    # Inject fake game instance and state
    fake = FakeInstance()
    gs.game_instance = fake
    gs.gs = fake.state

    # Start worker and wait for UI callback
    gs._start_ai_worker()

    assert called.wait(1.0), "UI callback was not called by AI worker within timeout"

    # Stop worker and ensure thread exits
    gs._stop_ai_worker()
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
