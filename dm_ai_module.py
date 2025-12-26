"""
Compatibility shim loader for local development and tests.

This file historically acted as a pure-Python stub for the C++/pybind11
`dm_ai_module` extension. To allow building and using the real C++
extension without renaming files, attempt to load a compiled
extension (dm_ai_module.* -> .pyd/.so) from this directory and, if
present, re-export its symbols. Otherwise fall back to the existing
pure-Python test stubs provided below.

This enables the test harness to use the real bindings after building
the C++ extension while keeping a pure-Python fallback for quick runs.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType

# Try to find a compiled extension in the same directory (Windows: .pyd, Unix: .so)
_here = os.path.dirname(__file__)
_candidates = [
    os.path.join(_here, "dm_ai_module.pyd"),
    os.path.join(_here, "dm_ai_module.so"),
    os.path.join(_here, "dm_ai_module.dll")
]
_ext_path = None
for p in _candidates:
    if os.path.exists(p):
        _ext_path = p
        break

if _ext_path:
    # Load the extension under a private name and re-export its public attrs
    try:
        loader = importlib.machinery.ExtensionFileLoader("_dm_ai_module_ext", _ext_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        # Copy public attributes into this module's globals so `import dm_ai_module` works
        for _k in dir(mod):
            if not _k.startswith("_"):
                globals()[_k] = getattr(mod, _k)
        # Also keep a reference to the loaded extension
        _compiled_module = mod
        global __all__
        __all__ = [name for name in dir(mod) if not name.startswith("_")]
    except Exception:
        # Fall through to Python stub below on any failure
        _compiled_module = None
else:
    _compiled_module = None

if _compiled_module is None:
    # --- Begin pure-Python fallback stubs (kept from original file) ---
    import json
    from types import SimpleNamespace
    from enum import Enum, IntEnum
    from dataclasses import dataclass, field
    from typing import Any, List, Dict, Optional

    class Civilization(Enum):
        FIRE = 'FIRE'

    class CardType(Enum):
        CREATURE = 'CREATURE'

    class Zone(Enum):
        HAND = 'HAND'
        DECK = 'DECK'
        MANA_ZONE = 'MANA_ZONE'
        GRAVEYARD = 'GRAVEYARD'
        BATTLE = 'BATTLE'
        MANA = 'MANA'

    class Phase(IntEnum):
        MAIN = 0
        ATTACK = 1

    class CommandType(Enum):
        TRANSITION = 'TRANSITION'
        MUTATE = 'MUTATE'
        FLOW = 'FLOW'
        QUERY = 'QUERY'
        DRAW_CARD = 'DRAW_CARD'
        DISCARD = 'DISCARD'
        DESTROY = 'DESTROY'
        MANA_CHARGE = 'MANA_CHARGE'
        TAP = 'TAP'
        UNTAP = 'UNTAP'
        POWER_MOD = 'POWER_MOD'
        ADD_KEYWORD = 'ADD_KEYWORD'
        RETURN_TO_HAND = 'RETURN_TO_HAND'
        BREAK_SHIELD = 'BREAK_SHIELD'
        SEARCH_DECK = 'SEARCH_DECK'
        SHIELD_TRIGGER = 'SHIELD_TRIGGER'
        NONE = 'NONE'

    def _zone_name(zone: Any) -> str:
        if isinstance(zone, Zone):
            if zone in (Zone.MANA, Zone.MANA_ZONE):
                return 'mana_zone'
            if zone == Zone.HAND:
                return 'hand'
            if zone == Zone.BATTLE:
                return 'battle'
            if zone == Zone.DECK:
                return 'deck'
        if isinstance(zone, str):
            s = zone.upper()
            if 'MANA' in s:
                return 'mana_zone'
            if 'HAND' in s:
                return 'hand'
            if 'BATTLE' in s:
                return 'battle'
            if 'DECK' in s:
                return 'deck'
        return 'hand'

    @dataclass
    class GameCommand:
        type: str
        params: Dict[str, Any] = field(default_factory=dict)

    class GameState:
        def __init__(self, capacity: int = 0):
            self.capacity = capacity
            self.active_modifiers: List[Any] = []
            self.players: List[Dict[str, Any]] = []
            self.turn_number: int = 0
            self.current_phase: Optional[Phase] = None

        def add_modifier(self, m: Any):
            self.active_modifiers.append(m)

        def _ensure_player(self, player_id: int):
            while len(self.players) <= player_id:
                ns = SimpleNamespace()
                ns.hand = []
                ns.mana_zone = []
                ns.battle = []
                ns.battle_zone = ns.battle
                ns.deck = []
                ns.graveyard = []
                self.players.append(ns)

        def add_card_to_hand(self, player_id: int, card_id: int, instance_id: int):
            self._ensure_player(player_id)
            ci = SimpleNamespace()
            ci.card_id = card_id
            ci.instance_id = instance_id
            ci.is_tapped = False
            self.players[player_id].hand.append(ci)

        def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, is_tapped: bool, _=None):
            self._ensure_player(player_id)
            ci = SimpleNamespace()
            ci.card_id = card_id
            ci.instance_id = instance_id
            ci.is_tapped = is_tapped
            self.players[player_id].battle.append(ci)
            self.players[player_id].battle_zone = self.players[player_id].battle

        def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int):
            self._ensure_player(player_id)
            ci = SimpleNamespace()
            ci.card_id = card_id
            ci.instance_id = instance_id
            self.players[player_id].deck.append(ci)

        def add_card_to_mana(self, player_id: int, card_id: int, instance_id: int):
            """Test helper: add a card instance to the player's mana zone."""
            self._ensure_player(player_id)
            ci = SimpleNamespace()
            ci.card_id = card_id
            ci.instance_id = instance_id
            # represent civilizations optionally on the instance
            ci.civilizations = []
            self.players[player_id].mana_zone.append(ci)

        def get_card_instance(self, instance_id: int):
            for p in self.players:
                for zone in ('hand', 'mana_zone', 'battle'):
                    for ci in getattr(p, zone, []):
                        if getattr(ci, 'instance_id', None) == instance_id:
                            return ci
            return None

        def execute_command(self, cmd: Any):
            if hasattr(cmd, 'execute'):
                cmd.execute(self)
                return
            # minimal fallback implementations omitted for brevity

        waiting_for_user_input: bool = False

    @dataclass
    class CardDefinition:
        id: int
        name: str = ""
        cost: int = 0

    # Minimal stubs for other classes used by tests
    class EffectSystem:
        pass

    class ActionDef:
        pass

    class EffectActionType(Enum):
        TRANSITION = 'TRANSITION'
        DRAW_CARD = 'DRAW_CARD'
        GET_GAME_STAT = 'GET_GAME_STAT'

    @dataclass
    class EffectDef:
        type: EffectActionType = EffectActionType.TRANSITION
        value1: Optional[int] = None
        value2: Optional[int] = None
        str_val: Optional[str] = None
        filter: Optional[Dict[str, Any]] = None
        condition: Optional[Dict[str, Any]] = None

    class CardKeywords:
        pass

    @dataclass
    class CardData:
        id: int
        name: str
        cost: int
        civilization: str
        power: int
        type: str
        keywords: List[Any]
        effects: List[Any]

    class CardRegistry:
        pass

    class TargetScope(Enum):
        NONE = 'NONE'
        PLAYER_SELF = 'PLAYER_SELF'
        PLAYER_OPPONENT = 'PLAYER_OPPONENT'
        TARGET_SELECT = 'TARGET_SELECT'

    class GenericCardSystem:
        @staticmethod
        def resolve_action(state: GameState, action_def, source_id: int = -1):
            # Minimal resolver for test purposes: support DRAW_CARD semantics
            try:
                if getattr(action_def, 'type', None) == EffectActionType.DRAW_CARD:
                    count = int(getattr(action_def, 'value1', 1) or 1)
                    pid = getattr(action_def, 'player_id', 0)
                    for _ in range(count):
                        if pid >= len(state.players):
                            continue
                        deck = getattr(state.players[pid], 'deck', [])
                        if not deck:
                            continue
                        ci = deck.pop()
                        state.players[pid].hand.append(ci)
            except Exception:
                pass

        @staticmethod
        def resolve_effect(state: GameState, eff: EffectDef, source_id: int = -1):
            """Resolve a simple EffectDef for tests.

            Supports GET_GAME_STAT with key 'MANA_CIVILIZATION_COUNT'.
            Returns computed value for convenience.
            """
            try:
                if getattr(eff, 'type', None) == EffectActionType.GET_GAME_STAT:
                    key = getattr(eff, 'str_val', None) or ''
                    if key == 'MANA_CIVILIZATION_COUNT':
                        # default to player 0 if no source provided
                        pid = 0
                        if state.players and pid < len(state.players):
                            civs = set()
                            for ci in getattr(state.players[pid], 'mana_zone', []):
                                # instance may carry 'civilizations' or card_id lookup may be needed
                                if hasattr(ci, 'civilizations') and ci.civilizations:
                                    for c in ci.civilizations:
                                        civs.add(c)
                            return len(civs)
                # default: no-op
                return None
            except Exception:
                return None

    class TensorConverter:
        INPUT_SIZE = 16

        @staticmethod
        def convert_to_tensor(game_state: GameState, player_id: int, card_db: Dict[int, Any]):
            vec = [0.0] * TensorConverter.INPUT_SIZE
            try:
                p = game_state.players[player_id]
                vec[0] = float(len(getattr(p, 'hand', [])))
                vec[1] = float(len(getattr(p, 'mana_zone', [])))
                vec[2] = float(len(getattr(p, 'battle', [])))
            except Exception:
                pass
            return vec

    def register_card_data(*args, **kwargs):
        return None

    # End of pure-Python fallback

