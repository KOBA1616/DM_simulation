import pytest
from dm_toolkit.gui import headless

@pytest.mark.slow
def test_headless_smoke_runs():
    """Run the smoke test logic directly using headless module."""
    sess = headless.create_session(p0_human=True)
    assert sess.gs is not None, "No GameState available"

    # Minimal logging callback to verify it can be attached
    sess.callback_log = lambda m: None
    sess.callback_update_ui = lambda: None

    # Check P0 hand
    p0 = sess.gs.players[0]
    hand = getattr(p0, 'hand', [])
    hand_ids = [getattr(c, 'instance_id', getattr(c, 'id', None)) for c in hand]

    if not hand_ids:
        # Step phase to populate/draw
        sess.step_phase()
        hand = getattr(p0, 'hand', [])
        hand_ids = [getattr(c, 'instance_id', getattr(c, 'id', None)) for c in hand]

    # If we have cards, try to find legal commands
    if hand_ids:
        iid = hand_ids[0]
        cmds = headless.find_legal_commands_for_instance(sess, iid)
        # We don't strictly assert cmds is non-empty because it depends on game state (mana etc)
        # But calling it should not crash.

        # Try to play it
        headless.play_instance(sess, iid)

    # Run some steps
    steps, over = headless.run_steps(sess, max_steps=50)
    assert steps >= 0
