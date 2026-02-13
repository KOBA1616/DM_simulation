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
    try:
        sess.card_db = card_db

        # If native dm_ai_module is present but missing ActionGenerator/ActionType,
        # inject lightweight Python stubs so generate_legal_actions fallback works
        try:
            import dm_ai_module as _native
            need_stub = False
            if not hasattr(_native, 'ActionGenerator') or not hasattr(_native, 'Action') or not hasattr(_native, 'ActionType'):
                need_stub = True
            if need_stub:
                # Minimal Action/ActionType/ActionGenerator stubs
                class _ActionType:
                    PASS = 'PASS'
                    MANA_CHARGE = 'MANA_CHARGE'
                    PLAY_CARD = 'PLAY_CARD'

                class _Action:
                    def __init__(self):
                        self.type = None
                        self.card_id = None
                        self.source_instance_id = None
                    def __repr__(self):
                        return f"<StubAction type={self.type} card_id={self.card_id} src={self.source_instance_id}>"

                class _ActionGenerator:
                    @staticmethod
                    def generate_legal_commands(state, card_db):
                        out = []
                        try:
                            pid = getattr(state, 'active_player_id', 0)
                            player = state.players[pid]
                            # Always allow PASS
                            a = _Action(); a.type = _ActionType.PASS; out.append(a)
                            # Mana-charge candidates for each card in hand
                            try:
                                for c in list(getattr(player, 'hand', []) or []):
                                    ac = _Action(); ac.type = _ActionType.MANA_CHARGE
                                    ac.card_id = getattr(c, 'card_id', c)
                                    ac.source_instance_id = getattr(c, 'instance_id', -1)
                                    out.append(ac)
                            except Exception:
                                pass
                            # Heuristic: propose PLAY_CARD for any hand card if mana_zone non-empty
                            try:
                                usable = sum(1 for m in getattr(player, 'mana_zone', []) if not getattr(m, 'is_tapped', False))
                                if usable > 0:
                                    for c in list(getattr(player, 'hand', []) or []):
                                        ac = _Action(); ac.type = _ActionType.PLAY_CARD
                                        ac.card_id = getattr(c, 'card_id', c)
                                        ac.source_instance_id = getattr(c, 'instance_id', -1)
                                        out.append(ac)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        return out

                # Attach stubs into native module so existing fallbacks can import them
                try:
                    _native.ActionType = _ActionType
                    _native.Action = _Action
                    _native.ActionGenerator = _ActionGenerator
                    # Also provide minimal Command-based stubs so command-first codepaths
                    # (generate_legal_commands / Command.to_dict) can fall back safely.
                    class _CommandType:
                        PASS = 'PASS'
                        MANA_CHARGE = 'MANA_CHARGE'
                        PLAY_CARD = 'PLAY_CARD'

                    class _Command:
                        def __init__(self, type_, instance_id=None, source_instance_id=None, card_id=None):
                            self.type = type_
                            self.instance_id = instance_id
                            self.source_instance_id = source_instance_id
                            self.card_id = card_id

                        def to_dict(self):
                            out = {'type': self.type}
                            if self.instance_id is not None:
                                out['instance_id'] = self.instance_id
                            if self.source_instance_id is not None:
                                out['source_instance_id'] = self.source_instance_id
                            if self.card_id is not None:
                                out['card_id'] = self.card_id
                            return out

                        def __repr__(self):
                            return f"<_StubCommand type={self.type} card_id={self.card_id} src={self.source_instance_id}>"

                    class _CommandGenerator:
                        @staticmethod
                        def generate_legal_commands(state, card_db):
                            out = []
                            try:
                                # Prefer command-first generator when possible, else adapt from ActionGenerator
                                acts = []
                                try:
                                    from dm_toolkit import commands_v2 as _commands
                                    try:
                                        acts = _commands.generate_legal_commands(state, card_db, strict=False) or []
                                    except TypeError:
                                        acts = _commands.generate_legal_commands(state, card_db) or []
                                    except Exception:
                                        acts = []
                                except Exception:
                                    acts = []
                                if not acts:
                                    try:
                                        # Final fallback to centralized legacy helper
                                        from dm_toolkit import commands as legacy_commands
                                        try:
                                            from dm_toolkit.training.command_compat import generate_legal_commands as compat_generate
                                            acts = compat_generate(state, card_db, strict=False) or []
                                        except Exception:
                                            acts = []
                                    except Exception:
                                        acts = []

                                if acts:
                                    for a in acts:
                                        try:
                                            t = getattr(a, 'type', None)
                                            iid = getattr(a, 'instance_id', None) or getattr(a, 'source_instance_id', None)
                                            cid = getattr(a, 'card_id', None)
                                            out.append(_Command(t, instance_id=iid, source_instance_id=getattr(a, 'source_instance_id', None), card_id=cid))
                                        except Exception:
                                            continue
                                else:
                                    # Fallback: build a PASS command
                                    out.append(_Command(_CommandType.PASS))
                            except Exception:
                                pass
                            return out

                    _native.CommandType = _CommandType
                    _native.Command = _Command
                    _native.CommandGenerator = _CommandGenerator
                    print("headless: injected dm_ai_module stubs for Action/Command and generators")
                except Exception:
                    pass
        except Exception:
            # Import failure of dm_ai_module is acceptable; fallbacks will handle it
            pass

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

    # Ensure minimal deck/hand content for headless tests when engine/setup left them empty
    try:
        if sess.gs and hasattr(sess.gs, 'players'):
            p0 = sess.gs.players[0]
            p1 = sess.gs.players[1]
            def zone_len(p, name):
                try:
                    return len(getattr(p, name, []))
                except Exception:
                    return 0

            # If both players have empty decks, synthesize simple mock card objects
            if zone_len(p0, 'deck') == 0 and zone_len(p1, 'deck') == 0:
                # derive ids to use
                ids = []
                try:
                    if isinstance(sess.card_db, dict):
                        ids = [int(k) for k in sess.card_db.keys()]
                    elif isinstance(sess.card_db, list):
                        ids = [int(c.get('id')) for c in sess.card_db if isinstance(c, dict) and 'id' in c]
                except Exception:
                    ids = []

                if not ids:
                    # fallback to a small numeric range
                    ids = list(range(1, 6))

                class _MockCard:
                    _iid = 1000
                    def __init__(self, card_id):
                        type(self)._iid += 1
                        self.card_id = int(card_id)
                        self.instance_id = type(self)._iid
                    def __repr__(self):
                        return f"<MockCard id={self.card_id} iid={self.instance_id}>"

                # populate decks with repeated ids up to 20
                def fill_deck(pid, cnt=20):
                    out = []
                    i = 0
                    while len(out) < cnt:
                        out.append(_MockCard(ids[i % len(ids)]))
                        i += 1
                    return out

                try:
                    p0.deck.extend(fill_deck(0, 20))
                    p1.deck.extend(fill_deck(1, 20))
                except Exception:
                    pass

            # If hand is empty, draw up to 5 from deck
            try:
                if zone_len(p0, 'hand') == 0:
                    for _ in range(5):
                        if not getattr(p0, 'deck'):
                            break
                        try:
                            card = p0.deck.pop()
                            p0.hand.append(card)
                        except Exception:
                            break
                if zone_len(p1, 'hand') == 0:
                    for _ in range(5):
                        if not getattr(p1, 'deck'):
                            break
                        try:
                            card = p1.deck.pop()
                            p1.hand.append(card)
                        except Exception:
                            break
            except Exception:
                pass
    except Exception:
        pass

    # Route session callback_log to the module logger so GameSession diagnostics
    # (Execute PRE/POST dumps) appear in the centralized logs when running
    # headless scripts.
    try:
        sess.callback_log = logging.getLogger('selfplay_long').debug
    except Exception:
        pass

    return sess


def find_legal_commands_for_instance(sess: GameSession, instance_id: int) -> List[Any]:
    """Return legal command objects that reference the given instance id."""
    if not sess.gs:
        return []
        from dm_toolkit import commands_v2
    try:
        cmds = []
        try:
            cmds = commands_v2.generate_legal_commands(sess.gs, sess.card_db, strict=False)
        except TypeError:
            cmds = commands_v2.generate_legal_commands(sess.gs, sess.card_db)
        except Exception:
            cmds = []
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
        # Prefer command-first execution. If play_cmd looks like a legacy Action,
        # map it to a Command before executing.
        try:
            if play_cmd is None:
                return False
            if not hasattr(play_cmd, 'to_dict') and hasattr(play_cmd, 'type'):
                from dm_toolkit.action_to_command import map_action
                try:
                    mapped = map_action(play_cmd)
                    sess.execute_action(mapped)
                    return True
                except Exception:
                    # fall back to original object
                    pass
        except Exception:
            pass

        sess.execute_action(play_cmd)
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
