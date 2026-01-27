"""Simple Python `CommandEncoder` used for prototype parity testing.

Index schema (matches existing `index_to_command.py` fallback):
- 0: PASS
- 1..19: MANA_CHARGE with `slot_index` equal to index
- 20..(20+PLAY_FROM_ZONE_SLOTS-1): PLAY_FROM_ZONE with `slot_index = index - 20`

This encoder provides `index_to_command` and `command_to_index` and a `TOTAL_COMMAND_SIZE` constant.
"""
from typing import Dict

class CommandEncoder:
    PASS_INDEX = 0
    MANA_CHARGE_BASE = 1
    MANA_CHARGE_SLOTS = 19  # indices 1..19 inclusive
    PLAY_FROM_ZONE_BASE = MANA_CHARGE_BASE + MANA_CHARGE_SLOTS  # 20
    PLAY_FROM_ZONE_SLOTS = 256

    TOTAL_COMMAND_SIZE = PLAY_FROM_ZONE_BASE + PLAY_FROM_ZONE_SLOTS

    @staticmethod
    def index_to_command(idx: int) -> Dict:
        if idx == CommandEncoder.PASS_INDEX:
            return {"type": "PASS"}
        if CommandEncoder.MANA_CHARGE_BASE <= idx < CommandEncoder.PLAY_FROM_ZONE_BASE:
            return {"type": "MANA_CHARGE", "slot_index": int(idx)}
        if idx >= CommandEncoder.PLAY_FROM_ZONE_BASE and idx < CommandEncoder.TOTAL_COMMAND_SIZE:
            return {"type": "PLAY_FROM_ZONE", "slot_index": int(idx - CommandEncoder.PLAY_FROM_ZONE_BASE)}
        raise IndexError(f"index out of range: {idx}")

    @staticmethod
    def command_to_index(cmd: Dict) -> int:
        t = cmd.get("type")
        if t == "PASS":
            return CommandEncoder.PASS_INDEX
        if t == "MANA_CHARGE":
            si = int(cmd.get("slot_index"))
            # validate range
            if si < CommandEncoder.MANA_CHARGE_BASE or si >= CommandEncoder.PLAY_FROM_ZONE_BASE:
                raise ValueError("slot_index out of range for MANA_CHARGE")
            return si
        if t == "PLAY_FROM_ZONE":
            si = int(cmd.get("slot_index"))
            if si < 0 or si >= CommandEncoder.PLAY_FROM_ZONE_SLOTS:
                raise ValueError("slot_index out of range for PLAY_FROM_ZONE")
            return CommandEncoder.PLAY_FROM_ZONE_BASE + si
        raise ValueError(f"unsupported command type: {t}")


# convenience wrappers to match existing naming
def index_to_command(idx: int) -> Dict:
    return CommandEncoder.index_to_command(int(idx))


def command_to_index(cmd: Dict) -> int:
    return CommandEncoder.command_to_index(cmd)
