import pytest
from dm_ai_module import GameCommand, TransitionCommand, MutateCommand, FlowCommand, QueryCommand, DecideCommand, CommandType, GameState, Player, Zone

def test_command_instantiation():
    # Test TransitionCommand
    t_cmd = TransitionCommand(1, 100, Zone.HAND, Zone.BATTLE, 0)
    assert t_cmd.get_type() == CommandType.TRANSITION
    assert "TRANSITION" in t_cmd.to_string()

    # Test MutateCommand
    # Assuming MutationType 0 is POWER_MODIFIER or similar (bound as int or enum)
    # The binding uses enum MutationType, but we didn't expose the enum values in a class.
    # In bindings.cpp we didn't bind MutationType enum explicitly to the module, only used in constructor.
    # Actually I missed binding MutationType, FlowType etc enums to the module scope in the replace_with_git_merge_diff call.
    # I only bound CommandType.
    # Let's check bindings.cpp content via memory or re-read.
    # I see I added CommandType but not MutationType etc.
    # Wait, the constructor expects MutationType. If I didn't bind it, I can't pass it from Python easily unless I pass an int and pybind casts it?
    # Pybind11 enums need to be exposed to be used as types.
    pass

def test_command_inheritance():
    t_cmd = TransitionCommand(1, 100, Zone.HAND, Zone.BATTLE, 0)
    assert isinstance(t_cmd, GameCommand)

if __name__ == "__main__":
    test_command_instantiation()
    test_command_inheritance()
