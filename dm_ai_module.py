"""Project-local dm_ai_module loader.

This repository builds a native extension module named `dm_ai_module`.
On Windows the built artifact is typically a `.pyd` under `bin/` or a CMake
build output directory (e.g. `build*/Release/`).

To keep imports consistent across GUI / scripts / tests, this file acts as the
canonical import target for `import dm_ai_module`.

- If a native extension is found, it is loaded in-place as the `dm_ai_module`
  module (so its init symbol name matches).
- If not found, we fall back to a lightweight pure-Python stub implementation
  sufficient for running unit tests and type-checking.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os
import sys
from enum import Enum
from types import ModuleType
from typing import Any, Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_native_paths(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "bin", "dm_ai_module*.pyd"),
        os.path.join(root, "build*", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build*", "**", "dm_ai_module*.so"),
        os.path.join(root, "build*", "**", "dm_ai_module*.dylib"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    # Prefer Release artifacts when multiple exist.
    def _score(p: str) -> tuple[int, int]:
        p_norm = p.replace("/", "\\").lower()
        return (
            0 if "\\release\\" in p_norm else 1,
            0 if "\\bin\\" in p_norm else 1,
        )

    uniq = sorted({os.path.normpath(p) for p in paths}, key=_score)
    return uniq


def _load_native_in_place(module_name: str, path: str) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, path)
    spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
    if spec is None:
        raise ImportError(f"Could not create spec for native module at: {path}")
    module = importlib.util.module_from_spec(spec)

    # Pre-register to support recursive imports during module init.
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _try_load_native() -> Optional[ModuleType]:
    root = _repo_root()

    # Allow explicit override (useful for debugging).
    override = os.environ.get("DM_AI_MODULE_NATIVE")
    candidates = [override] if override else _candidate_native_paths(root)

    for p in candidates:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                return _load_native_in_place(__name__, p)
            except Exception:
                # Try next candidate.
                continue
    return None


_native = _try_load_native()

if _native is not None:
    # Expose native symbols from this module.
    globals().update(_native.__dict__)

    # Minimal shims for compatibility when native builds lack some helpers.
    # Keep these intentionally small; higher-level compatibility belongs in dm_toolkit.
    if "DeclarePlayCommand" not in globals():

        class DeclarePlayCommand:  # type: ignore
            def __init__(self, player_id: int, card_id: int, source_instance_id: int):
                self.player_id = player_id
                self.card_id = card_id
                self.source_instance_id = source_instance_id

            def execute(self, state: Any) -> None:
                setattr(state, "_last_declared_play", {
                    "player_id": self.player_id,
                    "card_id": self.card_id,
                    "source_instance_id": self.source_instance_id,
                })

        globals()["DeclarePlayCommand"] = DeclarePlayCommand

    if "PayCostCommand" not in globals():

        class PayCostCommand:  # type: ignore
            def __init__(self, player_id: int, amount: int):
                self.player_id = player_id
                self.amount = amount

            def execute(self, state: Any) -> bool:
                # Best-effort stub. Real behavior is in native module.
                return True

        globals()["PayCostCommand"] = PayCostCommand

    if "ResolvePlayCommand" not in globals():

        class ResolvePlayCommand:  # type: ignore
            def __init__(self, player_id: int, card_id: int, card_def: Any = None):
                self.player_id = player_id
                self.card_id = card_id
                self.card_def = card_def

            def execute(self, state: Any) -> None:
                return

        globals()["ResolvePlayCommand"] = ResolvePlayCommand

else:
    # -----------------
    # Pure-Python fallback (tests / lint)
    # -----------------

    class Civilization(Enum):
        FIRE = 1
        WATER = 2
        NATURE = 3
        LIGHT = 4
        DARKNESS = 5


    class CardType(Enum):
        CREATURE = 1
        SPELL = 2


    class CardKeywords(int):
        pass


    class PassiveType(Enum):
        NONE = 0


    class PassiveEffect:
        def __init__(self, *args: Any, **kwargs: Any):
            pass


    class FilterDef(dict):
        pass


    class PlayerStub:
        def __init__(self) -> None:
            self.hand: list[Any] = []
            self.deck: list[Any] = []
            self.battle_zone: list[Any] = []
            self.graveyard: list[Any] = []
            self.mana_zone: list[Any] = []
            self.shield_zone: list[Any] = []


    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            self.turn_number = 0
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0

        def setup_test_duel(self) -> None:
            return


    class CardDefinition:
        def __init__(self, *args: Any, **kwargs: Any):
            pass


    _batch_callback: Optional[Any] = None


    def set_batch_callback(cb: Any) -> None:
        global _batch_callback
        _batch_callback = cb


    def has_batch_callback() -> bool:
        return _batch_callback is not None


    def clear_batch_callback() -> None:
        global _batch_callback
        _batch_callback = None


    class ActionEncoder:
        TOTAL_ACTION_SIZE = 10


    class NeuralEvaluator:
        def __init__(self, card_db: Any):
            self.card_db = card_db

        def evaluate(self, batch: list[Any]):
            if _batch_callback is not None:
                return _batch_callback(batch)
            policies = [[0.0] * ActionEncoder.TOTAL_ACTION_SIZE for _ in batch]
            values = [0.0 for _ in batch]
            return policies, values


    class CommandDef:
        def __init__(self, *args: Any, **kwargs: Any):
            pass


    class CardData:
        def __init__(self, *args: Any, **kwargs: Any):
            pass


    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, *args: Any, **kwargs: Any) -> None:
            return


    class EffectResolver:
        @staticmethod
        def resolve_action(state: Any, action: Any, card_db: Any) -> None:
            return


    class PhaseManager:
        @staticmethod
        def next_phase(state: Any, card_db: Any) -> None:
            return


    def register_card_data(data: Any) -> None:
        return


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


    class TargetScope(Enum):
        PLAYER_SELF = 1
        SELF = 1


    class Zone(Enum):
        DECK = 1
        HAND = 2
        GRAVEYARD = 3
        MANA = 4
        BATTLE_ZONE = 5
        SHIELD_ZONE = 6
