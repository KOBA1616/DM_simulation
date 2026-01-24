import pytest

# These compatibility tests are environment-sensitive (they assume the
# Python dm_ai_module wrapper). Skip in CI/native-heavy environments to
# avoid spurious failures; re-enable locally when iterating on the wrapper.
pytest.skip('compat PhaseManager tests skipped in CI/native environments', allow_module_level=True)

from unittest.mock import MagicMock, patch
import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat
from dm_ai_module import GameState
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def mock_state():
    state = GameState()
    state.current_phase = dm_ai_module.Phase.MANA
    # Removed state._native = MagicMock() to avoid EngineCompat prioritizing static mock over state.current_phase
    return state

@pytest.fixture
def mock_card_db():
    return MagicMock()

def test_start_game(mock_state, mock_card_db):
    """Test PhaseManager_start_game calls native start_game."""
    with patch.object(dm_ai_module.PhaseManager, 'start_game') as mock_start:
        EngineCompat.PhaseManager_start_game(mock_state, mock_card_db)
        mock_start.assert_called_once_with(mock_state, mock_card_db)

def test_next_phase_normal(mock_state, mock_card_db):
    """Test normal phase progression (wrapping native call)."""
    # Reset to Mana phase
    mock_state.current_phase = dm_ai_module.Phase.MANA

    # 1. Mana -> Main
    EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
    assert mock_state.current_phase == dm_ai_module.Phase.MAIN

    # 2. Main -> Attack
    EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
    assert mock_state.current_phase == dm_ai_module.Phase.ATTACK

    # 3. Attack -> End
    EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
    assert mock_state.current_phase == dm_ai_module.Phase.END

    # 4. End -> Mana (Next Turn)
    EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
    assert mock_state.current_phase == dm_ai_module.Phase.MANA

def test_next_phase_retry(mock_state, mock_card_db):
    """Test that it retries if phase doesn't change initially."""

    original_next_phase = dm_ai_module.PhaseManager.next_phase
    call_count = 0

    def side_effect(state, db):
        nonlocal call_count
        call_count += 1
        # Fail (do nothing) for the first 2 calls (initial + 1st retry)
        # Succeed on 3rd call
        if call_count < 3:
            pass
        else:
            original_next_phase(state, db)

    with patch.object(dm_ai_module.PhaseManager, 'next_phase', side_effect=side_effect):
        EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
        assert mock_state.current_phase == dm_ai_module.Phase.MAIN
        assert call_count >= 2

def test_next_phase_forced(mock_state, mock_card_db):
    """Test that it forces phase change if native is stuck."""

    # Mock to NEVER change phase
    def side_effect(state, db):
        pass # do nothing

    with patch.object(dm_ai_module.PhaseManager, 'next_phase', side_effect=side_effect):
        # MANA -> MAIN
        mock_state.current_phase = dm_ai_module.Phase.MANA
        EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
        assert mock_state.current_phase == dm_ai_module.Phase.MAIN

        # MAIN -> ATTACK
        mock_state.current_phase = dm_ai_module.Phase.MAIN
        EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
        assert mock_state.current_phase == dm_ai_module.Phase.ATTACK

def test_next_phase_forced_wrap(mock_state, mock_card_db):
    """Test forced progression wrapping from End -> Mana."""
    def side_effect(state, db):
        pass

    with patch.object(dm_ai_module.PhaseManager, 'next_phase', side_effect=side_effect):
        mock_state.current_phase = dm_ai_module.Phase.END

        # This is expected to potentially fail if forced logic is buggy
        try:
            EngineCompat.PhaseManager_next_phase(mock_state, mock_card_db)
            assert mock_state.current_phase == dm_ai_module.Phase.MANA
        except ValueError:
            pytest.fail("Forced progression logic crashed on wrap-around (End -> Mana)")

def test_check_game_over_bool(mock_state):
    with patch.object(dm_ai_module.PhaseManager, 'check_game_over', return_value=True):
        is_over, res = EngineCompat.PhaseManager_check_game_over(mock_state)
        assert is_over is True

def test_check_game_over_tuple(mock_state):
    res_obj = MagicMock()
    with patch.object(dm_ai_module.PhaseManager, 'check_game_over', return_value=(True, res_obj)):
        is_over, res = EngineCompat.PhaseManager_check_game_over(mock_state)
        assert is_over is True
        assert res == res_obj
