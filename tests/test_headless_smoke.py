import pytest
import dm_ai_module
from dm_toolkit.gui import headless

@pytest.mark.skipif(not getattr(dm_ai_module, 'IS_NATIVE', False), reason="Requires native engine")
@pytest.mark.slow
def test_headless_smoke_runs():
    """ヘッドレスモジュール経由のスモークテスト。

    再発防止: headless.find_legal_commands_for_instance の import バグ修正後も
              クラッシュなくリストを返すことをここで確認する。
    """
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
        # 再発防止: cmds は必ず list を返すこと（NameError で [] が返るバグが再発していないか確認）
        assert isinstance(cmds, list), (
            f"find_legal_commands_for_instance が list を返していない: {type(cmds)}\n"
            "再発防止: from dm_toolkit import commands を return [] の後に書かないこと"
        )

        # Try to play it (result is bool regardless of whether a command was found)
        result = headless.play_instance(sess, iid)
        assert isinstance(result, bool), (
            f"play_instance が bool を返していない: {type(result)}"
        )

    # Run some steps
    steps, over = headless.run_steps(sess, max_steps=50)
    assert steps >= 0
