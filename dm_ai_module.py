"""Minimal dm_ai_module fallback for tests.

Contains compact implementations of a few engine-facing types and helpers
used by tests and tooling. This intentionally keeps behavior simple and
well-structured to avoid import-time issues.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, List, Optional

IS_NATIVE = False


class CardStub:
    _iid = 1000

    def __init__(self, card_id: int, instance_id: Optional[int] = None):
        if instance_id is None:
            CardStub._iid += 1
            instance_id = CardStub._iid
        self.card_id = card_id
        self.instance_id = instance_id


class Player:
    def __init__(self, pid: int = 0):
        self.player_id = pid
        self.hand: List[CardStub] = []
        self.mana_zone: List[CardStub] = []
        self.battle_zone: List[CardStub] = []
        self.graveyard: List[CardStub] = []


class GameState:
    def __init__(self):
        self.players: List[Player] = [Player(0), Player(1)]
        self.active_player_id = 0

    def add_card_to_hand(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
        if instance_id is not None and count == 1:
            c = CardStub(card_id, instance_id)
            self.players[player].hand.append(c)
            return c
        for _ in range(count):
            c = CardStub(card_id)
            self.players[player].hand.append(c)
        return self.players[player].hand[-1]

    def add_card_to_mana(self, player: int, card_id: int, count: int = 1):
        for _ in range(count):
            self.players[player].mana_zone.append(CardStub(card_id))

    def get_next_instance_id(self) -> int:
        # Return a fresh instance id suitable for tests
        CardStub._iid += 1
        return CardStub._iid


class CommandType(IntEnum):
    NONE = 0
    PLAY_FROM_ZONE = 1
    MANA_CHARGE = 2
    TRANSITION = 3
    ATTACK = 4
    PASS = 5


class CommandDef:
    def __init__(self):
        self.type = CommandType.NONE
        self.amount = 0
        self.str_param = ''
        self.optional = False
        self.instance_id = 0
        self.target_instance = 0
        self.owner_id = 0
        self.from_zone = ''
        self.to_zone = ''
        self.mutation_kind = ''
        self.input_value_key = ''
        self.output_value_key = ''
        self.target_filter = None
        self.target_group = None


class GameInstance:
    def __init__(self, seed: int = 0, card_db: Any = None):
        self.state = GameState()

    def start_game(self):
        self.state.active_player_id = 0

    def execute_command(self, cmd: Any):
        # Support dict-based MANA_CHARGE: {'type': 'MANA_CHARGE', 'instance_id': X}
        try:
            if isinstance(cmd, dict) and cmd.get('type') in ('MANA_CHARGE',):
                pid = self.state.active_player_id
                iid = cmd.get('instance_id') or cmd.get('card_id') or 0
                self.state.players[pid].mana_zone.append(CardStub(iid))
        except Exception:
            pass


class CommandSystem:
    @staticmethod
    def execute_command(state: Any, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
        """Simple CommandSystem fallback handling common dict-style commands.

        Supports at least: {'type': 'MANA_CHARGE', 'instance_id': ...}
        """
        try:
            # Normalize type
            t = None
            if isinstance(cmd, dict):
                t = cmd.get('type')
            else:
                t = getattr(cmd, 'type', None)

            # Support numeric enum-like objects
            try:
                if hasattr(t, 'name'):
                    t = t.name
            except Exception:
                pass

            # Determine target player
            pid = getattr(state, 'active_player_id', player_id)

            if t in ("MANA_CHARGE", 'MANA_CHARGE', 2):
                iid = None
                if isinstance(cmd, dict):
                    iid = cmd.get('instance_id') or cmd.get('source_instance_id') or cmd.get('card_id')
                else:
                    iid = getattr(cmd, 'instance_id', None) or getattr(cmd, 'source_instance_id', None)

                # find by instance id in hand
                if iid is not None:
                    for c in list(getattr(state.players[pid], 'hand', [])):
                        try:
                            if getattr(c, 'instance_id', None) == iid:
                                try:
                                    state.players[pid].hand.remove(c)
                                except Exception:
                                    pass
                                try:
                                    state.players[pid].mana_zone.append(c)
                                except Exception:
                                    pass
                                return
                        except Exception:
                            continue

                # fallback: move first card from hand to mana_zone
                try:
                    hand = getattr(state.players[pid], 'hand', [])
                    if hand:
                        c = hand.pop(0)
                        try:
                            state.players[pid].mana_zone.append(c)
                        except Exception:
                            pass
                except Exception:
                    pass
                return
        except Exception:
            pass


class JsonLoader:
    @staticmethod
    def load_cards(path: str) -> Any:
        """Load a card DB JSON file and return the parsed object.

        Path is interpreted relative to the current working directory if not absolute.
        """
        import json
        import os

        p = path
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), path)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}


class PhaseManager:
    @staticmethod
    def start_game(state: GameState, card_db: Any = None) -> None:
        try:
            # Try to call instance method if present
            if hasattr(state, 'start_game') and callable(getattr(state, 'start_game')):
                try:
                    state.start_game()
                    return
                except Exception:
                    pass
            # Otherwise apply minimal defaults
            if hasattr(state, 'active_player_id'):
                setattr(state, 'active_player_id', 0)
        except Exception:
            pass


class DevTools:
    @staticmethod
    def move_cards(gs: GameState, key: Any, src: int, dst: int, count: int = 1, card_filter: int = -1) -> int:
        # key may be a player id or an instance id. Try instance lookup first.
        try:
            iid = int(key)
        except Exception:
            return 0

        # instance-id lookup
        for pid, p in enumerate(gs.players):
            for zone_name in ('hand', 'mana_zone', 'battle_zone', 'graveyard'):
                zone = getattr(p, zone_name, [])
                for i, c in enumerate(list(zone)):
                    if getattr(c, 'instance_id', None) == iid:
                        # move this instance to dst (dst: 1->hand,2->mana,3->battle,4->grave)
                        dst_map = {1: 'hand', 2: 'mana_zone', 3: 'battle_zone', 4: 'graveyard'}
                        dst_attr = dst_map.get(dst)
                        if dst_attr is None:
                            return 0
                        try:
                            getattr(p, zone_name).pop(i)
                        except Exception:
                            pass
                        try:
                            getattr(p, dst_attr).append(c)
                        except Exception:
                            pass
                        return 1

        # else treat key as player id and move `count` cards from src to dst
        try:
            pid = int(key)
            src_map = {0: 'deck', 1: 'hand', 2: 'mana_zone', 3: 'battle_zone', 4: 'graveyard'}
            dst_map = src_map
            sattr = src_map.get(src)
            dattr = dst_map.get(dst)
            if sattr is None or dattr is None:
                return 0
            moved = 0
            for _ in range(count):
                try:
                    c = getattr(gs.players[pid], sattr).pop(0)
                except Exception:
                    break
                try:
                    getattr(gs.players[pid], dattr).append(c)
                except Exception:
                    pass
                moved += 1
            return moved
        except Exception:
            return 0


__all__ = ['IS_NATIVE', 'GameInstance', 'GameState', 'Player', 'CardStub', 'DevTools', 'JsonLoader', 'PhaseManager', 'CommandSystem', 'CommandType', 'CommandDef']


