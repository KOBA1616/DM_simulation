"""Headless helpers for testing GUI-free game sessions.

Provides utilities to create and drive a `GameSession` without importing any
Qt dependencies so tests can run in headless CI environments.
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import logging

from dm_toolkit.gui.game_session import GameSession
from dm_toolkit.engine.compat import EngineCompat


def create_session(card_db: Optional[Dict[int, Any]] = None,
                   p0_deck: Optional[List[int]] = None,
                   p1_deck: Optional[List[int]] = None,
                   p0_human: bool = False,
                   p1_human: bool = False) -> GameSession:
    """Create and initialize a GameSession suitable for tests.

    - Does NOT import any PyQt modules.
    - Returns a ready-to-use `GameSession` where `initialize_game` or
      `reset_game` has been called and player modes set according to flags.

    再発防止: スタブ注入（CommandType/Command/CommandGenerator）は削除済み。
              native (IS_NATIVE=True) では IntentGenerator が常に利用可能。
              スタブは dm_ai_module グローバル汚染・print 出力汚染を招くため廃止。
    再発防止: _MockCard デッキ注入・手札注入フォールバックは削除済み。
              native の reset_game() がデッキ・手札を正しく初期化する。
    """
    sess = GameSession()
    # Set human/AI modes
    sess.set_player_mode(0, 'Human' if p0_human else 'AI')
    sess.set_player_mode(1, 'Human' if p1_human else 'AI')

    # Provide a minimal card_db if none given (EngineCompat will try to load)
    if card_db is None:
        # Prefer EngineCompat robust loader to keep behavior consistent
        try:
            card_db = EngineCompat.load_cards_robust(os.path.join('data', 'cards.json'))
        except Exception:
            card_db = {}

        # If loader failed or returned empty, try direct JSON read as a fallback
        if not card_db:
            try:
                import json
                p = os.path.join(os.getcwd(), 'data', 'cards.json')
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                    if isinstance(raw, dict):
                        card_db = raw
                    elif isinstance(raw, list):
                        card_db = {int(c.get('id')): c for c in raw if isinstance(c, dict) and 'id' in c}
            except Exception:
                card_db = {}

    # Initialize game state using provided DB and optional decks
    # 再発防止: CommandType/Command/CommandGenerator スタブ注入は削除済み。
    #           native (IS_NATIVE=True) では IntentGenerator が常に利用可能であり、
    #           スタブ注入は dm_ai_module グローバル状態汚染・print 出力汚染を招くため不要。
    try:
        sess.card_db = card_db

        # If no decks provided, try to construct simple decks from card_db
        if p0_deck is None and p1_deck is None:
            # Build minimal decks using available ids (repeat to reach 40)
            ids = []
            try:
                if isinstance(card_db, dict):
                    ids = [int(k) for k in card_db.keys()]
                elif isinstance(card_db, list):
                    ids = [int(c.get('id')) for c in card_db if isinstance(c, dict) and 'id' in c]
            except Exception:
                ids = []

            if ids:
                def make_deck(n=40):
                    out = []
                    i = 0
                    while len(out) < n:
                        out.append(ids[i % len(ids)])
                        i += 1
                    return out

                p0_deck = make_deck(40)
                p1_deck = make_deck(40)[::-1]

        if p0_deck is None and p1_deck is None:
            sess.initialize_game(card_db)
        else:
            sess.reset_game(p0_deck, p1_deck)
    except Exception:
        # fallback: still return a session; tests can manipulate sess.gs directly
        pass

    # 再発防止: _MockCard デッキ注入・手札注入フォールバックは削除済み。
    #           native (IS_NATIVE=True) では reset_game() がデッキ・手札を正しく設定するため不要。
    #           非 native 環境のテストは pytestmark skipif で除外済み。

    # Route session callback_log to the module logger so GameSession diagnostics
    # (Execute PRE/POST dumps) appear in the centralized logs when running
    # headless scripts.
    try:
        sess.callback_log = logging.getLogger('selfplay_long').debug
    except Exception:
        pass

    return sess


def find_legal_commands_for_instance(sess: GameSession, instance_id: int) -> List[Any]:
    """Return legal command objects that reference the given instance id.

    再発防止: import 文を return 文の後に置いてはならない（デッドコード＋NameError の原因）。
              from dm_toolkit import commands は早期 return の前に配置すること。
    """
    # 再発防止: import は早期 return より前に書くこと。後に書くと NameError になる。
    from dm_toolkit import commands

    if not sess.gs:
        return []
    try:
        # commands.generate_legal_commands は strict/skip_wrapper を受け付ける
        cmds = commands.generate_legal_commands(sess.gs, sess.card_db, skip_wrapper=True)
    except Exception:
        return []
    out = []
    for c in cmds:
        try:
            d = c.to_dict()
        except Exception:
            d = {}
        if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
            out.append(c)
    return out


def play_instance(sess: GameSession, instance_id: int) -> bool:
    """Find a PLAY_CARD (or other) legal command for `instance_id` and execute it.

    Returns True if a command was found and executed, False otherwise.
    """
    cmds = find_legal_commands_for_instance(sess, instance_id)
    if not cmds:
        return False

    # Prefer PLAY_CARD, otherwise take first
    play_cmd = None
    for c in cmds:
        try:
            d = c.to_dict()
        except Exception:
            d = {}
        if d.get('type') == 'PLAY_CARD':
            play_cmd = c
            break

    if play_cmd is None:
        play_cmd = cmds[0]

    try:
        sess.execute_command(play_cmd)
        return True
    except Exception:
        return False


def run_steps(sess: GameSession, max_steps: int = 1000) -> Tuple[int, bool]:
    """Step the session up to `max_steps` or until game_over.

    Returns (steps_taken, game_over_flag).
    """
    steps = 0
    while steps < max_steps:
        if sess.is_game_over():
            return steps, True
        sess.step_phase()
        steps += 1
    return steps, sess.is_game_over()
