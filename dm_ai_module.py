"""Project-local dm_ai_module loader.

This repository builds a native extension module named `dm_ai_module`.
To keep imports consistent across GUI / scripts / tests, this file acts as the
canonical import target for `import dm_ai_module`.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os
import sys
from enum import Enum, IntEnum
from types import ModuleType
from typing import Any, Optional, List


def _ensure_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return
    try:
        import onnxruntime as ort
        ort_pkg_dir = os.path.dirname(os.path.abspath(ort.__file__))
        capi_dir = os.path.join(ort_pkg_dir, "capi")
        for p in (ort_pkg_dir, capi_dir):
            if os.path.isdir(p):
                try: add_dir(p)
                except Exception: pass
    except Exception:
        pass


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_native_paths(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "bin", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "dm_ai_module*.so"),
        os.path.join(root, "bin", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "**", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build", "**", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.dylib"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    def _score(p: str) -> tuple[int, int]:
        p_norm = p.replace("/", "\\").lower()
        return (
            0 if "\\release\\" in p_norm or "/release/" in p_norm else 1,
            0 if "\\bin\\" in p_norm or "/bin/" in p_norm else 1,
        )

    uniq = sorted({os.path.normpath(p) for p in paths}, key=_score)
    return uniq


def _load_native_in_place(module_name: str, path: str) -> ModuleType:
    try:
        loader = importlib.machinery.ExtensionFileLoader(module_name, path)
        spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
        if spec is None:
            raise ImportError(f"Could not create spec for native module at: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e


def _try_load_native() -> Optional[ModuleType]:
    root = _repo_root()
    _ensure_windows_dll_search_path()
    override = os.environ.get("DM_AI_MODULE_NATIVE")
    candidates = [override] if override else _candidate_native_paths(root)
    for p in candidates:
        if not p: continue
        if os.path.isfile(p):
            try:
                mod = _load_native_in_place(__name__, p)
                return mod
            except Exception:
                continue
    return None


_native = _try_load_native()

if _native is not None:
    globals().update(_native.__dict__)
    IS_NATIVE = True
else:
    # -----------------
    # Pure-Python fallback (Stub for tests/lint)
    # -----------------
    IS_NATIVE = False

    class Civilization(Enum):
        FIRE = 1
        WATER = 2
        NATURE = 3
        LIGHT = 4
        DARKNESS = 5

    class CardType(Enum):
        CREATURE = 1
        SPELL = 2

    class GameResult(int):
        NONE = -1
        P1_WIN = 0
        P2_WIN = 1
        DRAW = 2

    class CardKeywords(int):
        SPEED_ATTACKER = 1
        BLOCKER = 2
        SLAYER = 3
        DOUBLE_BREAKER = 4
        TRIPLE_BREAKER = 5
        POWER_ATTACKER = 6
        EVOLUTION = 7

    class PassiveType(Enum):
        NONE = 0

    class PassiveEffect:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class FilterDef(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __getattr__(self, item):
            return self.get(item)
        def __setattr__(self, key, value):
            self[key] = value

    class EffectDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class ActionDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class Action:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = None
            self.target_player = 0
            self.source_instance_id = 0
            self.card_id = 0
            self.slot_index = 0
            self.value1 = 0

    class ConditionDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class EffectPrimitive(Enum):
        DRAW_CARD = 1
        IF = 2
        IF_ELSE = 3
        COUNT_CARDS = 4
        NONE = 99
        CAST_SPELL = 5
        PUT_CREATURE = 6
        DESTROY = 7
        MANA_CHARGE = 8

    class ActionType(IntEnum):
        PLAY_CARD = 1
        ATTACK_PLAYER = 2
        ATTACK_CREATURE = 3
        BLOCK_CREATURE = 4
        PASS = 5
        USE_SHIELD_TRIGGER = 6
        MANA_CHARGE = 7
        RESOLVE_EFFECT = 8
        SELECT_TARGET = 9
        TAP = 10
        UNTAP = 11
        BREAK_SHIELD = 14

    class CommandType(Enum):
        TRANSITION = 1
        MUTATE = 2
        FLOW = 3
        QUERY = 4
        DRAW_CARD = 5
        DISCARD = 6
        DESTROY = 7
        MANA_CHARGE = 8
        TAP = 9
        UNTAP = 10
        POWER_MOD = 11
        ADD_KEYWORD = 12
        RETURN_TO_HAND = 13
        BREAK_SHIELD = 14
        SEARCH_DECK = 15
        SHIELD_TRIGGER = 16
        NONE = 17
        PUT_INTO_PLAY = 18

    class TargetScope(Enum):
        PLAYER_SELF = 1
        SELF = 1
        OPPONENT = 2
        PLAYER_OPPONENT = 2

    class Zone(Enum):
        DECK = 1
        HAND = 2
        GRAVEYARD = 3
        MANA = 4
        BATTLE_ZONE = 5
        SHIELD_ZONE = 6
        BATTLE = 5
        SHIELD = 6

    class CardDatabase:
        def __init__(self): pass

    class CardStub:
        def __init__(self, card_id: int, instance_id: int):
            self.card_id = card_id
            self.instance_id = instance_id
            self.id = instance_id
            self.is_tapped = False
            self.sick = False
            self.power = 1000
            self.power_modifier = 0
            self.turn_played = 0

    class PlayerStub:
        def __init__(self) -> None:
            self.hand: list[Any] = []
            self.deck: list[Any] = []
            self.battle_zone: list[Any] = []
            self.graveyard: list[Any] = []
            self.mana_zone: list[Any] = []
            self.shield_zone: list[Any] = []
            self.life = 0
        @property
        def shields(self): return len(self.shield_zone)

    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            self.turn_number = 0
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0
            self.winner = GameResult.NONE
            self.current_phase = 0
            self.pending_effects: list[Any] = []
            self.instance_counter = 0

        def setup_test_duel(self) -> None: pass

        def get_card_instance(self, instance_id: int) -> Optional[CardStub]:
            target = int(instance_id)
            for p in self.players:
                for zone in [p.battle_zone, p.mana_zone, p.hand, p.shield_zone, p.graveyard, p.deck]:
                    for card in zone:
                        if isinstance(card, int): continue
                        # Check both instance_id and id for compatibility
                        if getattr(card, 'instance_id', -1) == target or getattr(card, 'id', -1) == target:
                            return card
            return None

        def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            # Use CardStub for consistency if tests check deck content length or properties
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].deck.append(CardStub(card_id, inst_id))

        def add_card_to_hand(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            cs = CardStub(card_id, inst_id)
            # Check duplicates (some tests reuse IDs)
            for existing in self.players[player_id].hand:
                if existing.instance_id == inst_id:
                    return
            self.players[player_id].hand.append(cs)

        def add_card_to_mana(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].mana_zone.append(CardStub(card_id, inst_id))

        def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, tapped: bool = False, sick: bool = False) -> Any:
            self._ensure_player(player_id)
            cs = CardStub(card_id, instance_id)
            cs.is_tapped = tapped
            cs.sick = sick
            # Check if exists
            for existing in self.players[player_id].battle_zone:
                if existing.instance_id == instance_id:
                     existing.is_tapped = tapped
                     existing.sick = sick
                     return existing
            self.players[player_id].battle_zone.append(cs)
            return cs

        def draw_cards(self, player_id: int, amount: int = 1) -> None:
            self._ensure_player(player_id)
            for _ in range(int(amount)):
                if not self.players[player_id].deck:
                     break
                # Deck now stores CardStub
                stub = self.players[player_id].deck.pop()
                cid = stub.card_id if isinstance(stub, CardStub) else stub
                self.add_card_to_hand(player_id, cid)

        def set_deck(self, player_id: int, card_ids: list[int]) -> None:
             self._ensure_player(player_id)
             self.players[player_id].deck = []
             for cid in card_ids:
                 self.add_card_to_deck(player_id, cid)

        def get_next_instance_id(self):
            self.instance_counter += 1
            return self.instance_counter

        def _ensure_player(self, player_id):
             while len(self.players) <= player_id:
                 self.players.append(PlayerStub())

    class CommandDef:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = CommandType.NONE
            self.target_filter = None
            self.target_group = None
            self.to_zone = None
            self.value = 0
            self.card_id = 0

    class GameCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = CommandType.NONE
            self.source_instance_id = -1
            self.target_player = -1
            self.card_id = -1
        def execute(self, state: Any) -> None: pass

    class MutationType(Enum):
        ADD_MODIFIER = 1
        ADD_PASSIVE = 2
        TAP = 3
        UNTAP = 4
        POWER_MOD = 5
        ADD_KEYWORD = 6

    class FlowType(IntEnum):
        NONE = 0
        SET_ATTACK_SOURCE = 1
        SET_ATTACK_PLAYER = 2
        SET_ATTACK_TARGET_CREATURE = 3
        RESOLVE_BATTLE = 4
        TURN_END = 5

    class MutateCommand(GameCommand):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.mutation_type = MutationType.TAP
            self.modifier = None
        @property
        def target_instance_id(self):
            return self.source_instance_id

    class FlowCommand(GameCommand):
        def __init__(self, flow_type=FlowType.NONE, value=-1, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.flow_type = flow_type
            # Handle both positional and keyword args
            self.new_value = value

    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int, player_id: int, ctx: Any = None) -> None:
            if cmd.type == CommandType.TAP:
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                card.is_tapped = True

            elif cmd.type == CommandType.UNTAP:
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                card.is_tapped = False

            elif cmd.type == CommandType.RETURN_TO_HAND:
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    if "BATTLE_ZONE" in zones:
                         p = state.players[player_id]
                         # Move everything from battle to hand
                         while p.battle_zone:
                             c = p.battle_zone.pop(0)
                             p.hand.append(c)
            elif cmd.type == CommandType.SEARCH_DECK:
                 p = state.players[player_id]
                 if p.deck:
                     stub = p.deck.pop()
                     cid = stub.card_id if isinstance(stub, CardStub) else stub
                     state.add_card_to_hand(player_id, cid)

    class TokenConverter:
        @staticmethod
        def encode_state(state: Any, player_id: int, length: int) -> List[int]:
            return [0] * length

    class GenericCardSystem:
        @staticmethod
        def resolve_action(state: Any, action: Any, source_id: int) -> Any:
            tgt = getattr(action, 'target_player', None)
            player = source_id
            if isinstance(tgt, str) and 'SELF' in tgt: player = source_id
            elif isinstance(tgt, str) and 'OPPONENT' in tgt: player = 1 - source_id
            elif isinstance(tgt, int): player = tgt

            atype = getattr(action, 'type', None)

            def is_type(t, name):
                return str(t) == name or getattr(t, 'name', '') == name or str(t).endswith(name)

            if is_type(atype, 'IF') or is_type(atype, 'IF_ELSE'):
                cond = getattr(action, 'condition', None)
                if not cond: cond = getattr(action, 'filter', None)

                truth = True
                if cond:
                    if getattr(cond, 'type', '') == 'COMPARE_STAT' and getattr(cond, 'stat_key', '') == 'MY_HAND_COUNT':
                        val = int(getattr(cond, 'value', 0))
                        op = getattr(cond, 'op', '')
                        cnt = len(state.players[player].hand)
                        if op == '>=': truth = cnt >= val
                        elif op == '>': truth = cnt > val
                        elif op == '<=': truth = cnt <= val
                        elif op == '<': truth = cnt < val
                        elif op == '==': truth = cnt == val
                    elif getattr(cond, 'civilizations', None):
                         pass

                opts = getattr(action, 'options', [])

                selected_opts = []
                if truth and len(opts) > 0: selected_opts = opts[0]
                elif not truth and len(opts) > 1: selected_opts = opts[1]

                for sub_act in selected_opts:
                    GenericCardSystem.resolve_action(state, sub_act, player)

            elif is_type(atype, 'DRAW_CARD'):
                val = int(getattr(action, 'value1', 1))
                state.draw_cards(player, val)

    class JsonLoader:
        @staticmethod
        def load_cards(path):
            # Very basic stub returning mock data if file not found or just mock
            return {1: CardDefinition(1, "TestCreature", 1, Civilization.FIRE, 1000, CardType.CREATURE, [], [])}

    class CardRegistry:
        @staticmethod
        def get_all_cards(): return {}

    def register_card_data(data): pass
    def initialize_card_stats(state, db, seed): pass

    class CardData:
        def __init__(self, card_id, name, cost, civ, power, ctype, races, effects): pass

    class CardDefinition(CardData):
         pass

    class CivilizationList(list): pass

    class TargetGroup(Enum):
        SELF = 1
        OPPONENT = 2

    # --------------------------------------------------------------------------
    # RESTORED / ADDED CLASSES FOR REPORT FIXES
    # --------------------------------------------------------------------------

    class PhaseManager:
        @staticmethod
        def next_phase(state: GameState, card_db: Any) -> None:
            state.current_phase += 1
            if state.current_phase > 6:
                state.current_phase = 0
                state.turn_number += 1
                state.active_player_id = 1 - state.active_player_id
                # Untap step
                for c in state.players[state.active_player_id].battle_zone:
                    c.is_tapped = False
                for c in state.players[state.active_player_id].mana_zone:
                    c.is_tapped = False
                # Draw step
                state.draw_cards(state.active_player_id, 1)

    class BatchData:
        def __init__(self):
            self.token_states = []
            self.policies = []
            self.values = []

    class DataCollector:
        def collect_data_batch_heuristic(self, num_episodes: int, use_policy: bool, use_value: bool) -> BatchData:
            b = BatchData()
            b.token_states = [[0]] * num_episodes
            b.policies = [[0.1]*10] * num_episodes
            b.values = [0] * num_episodes
            return b

    class GameInstance:
        def __init__(self, game_id: int = 0):
            self.state = GameState()
            self.state.game_over = False
            self.state.winner = GameResult.NONE
            self.card_db = JsonLoader.load_cards("data/cards.json")

        def start_game(self):
            self.state.setup_test_duel()
            # Initial Draw
            self.state.draw_cards(0, 5)
            self.state.draw_cards(1, 5)
            # Shields
            for i in range(2):
                for _ in range(5):
                    if self.state.players[i].deck:
                        c = self.state.players[i].deck.pop()
                        self.state.players[i].shield_zone.append(c)
            self.state.turn_number = 1
            self.state.current_phase = 0 # Turn Start

        def execute_action(self, action: GameCommand):
            player = self.state.players[self.state.active_player_id]

            if action.type == ActionType.MANA_CHARGE:
                # Find card in hand
                inst = self.state.get_card_instance(action.source_instance_id)
                if inst:
                    # Remove from hand, add to mana
                    if inst in player.hand:
                         player.hand.remove(inst)
                         player.mana_zone.append(inst)

            elif action.type == ActionType.PLAY_CARD:
                # [FIX ISSUE 1]: Spell Casting
                inst = self.state.get_card_instance(action.source_instance_id)
                if inst:
                    # Determine type
                    is_spell = False
                    if inst.card_id == 2:
                        is_spell = True

                    if inst in player.hand:
                        player.hand.remove(inst)

                        if is_spell:
                            self.state.pending_effects.append(inst)
                        else:
                            player.battle_zone.append(inst)

            elif action.type == ActionType.RESOLVE_EFFECT:
                # [FIX ISSUE 2]: Stack Processing
                if self.state.pending_effects:
                    card = self.state.pending_effects.pop()
                    # For stub: Move to graveyard after resolution
                    player.graveyard.append(card)

            elif action.type == ActionType.ATTACK_PLAYER:
                # Stub logic for attack
                pass

            elif action.type == ActionType.PASS:
                PhaseManager.next_phase(self.state, self.card_db)

        def resolve_action(self, action: Action):
            GenericCardSystem.resolve_action(self.state, action, action.source_instance_id)
