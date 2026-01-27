from native_prototypes.index_to_command.command_encoder import CommandEncoder, index_to_command, command_to_index
import pytest


def test_round_trip_sample_indices():
    # test a representative subset within TOTAL_COMMAND_SIZE
    caps = [0, 1, 5, 18, 19, 20, 21, 50, CommandEncoder.TOTAL_COMMAND_SIZE - 1]
    for i in caps:
        cmd = index_to_command(i)
        idx2 = command_to_index(cmd)
        assert idx2 == i


def test_command_to_index_invalid():
    with pytest.raises(ValueError):
        command_to_index({"type": "MANA_CHARGE", "slot_index": 0})
    with pytest.raises(ValueError):
        command_to_index({"type": "PLAY_FROM_ZONE", "slot_index": -1})
    with pytest.raises(ValueError):
        command_to_index({"type": "UNKNOWN"})
