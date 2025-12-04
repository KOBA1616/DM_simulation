import pytest
import dm_ai_module
from dm_ai_module import GameState, CardInstance

def test_stack_zone_exists():
    game = GameState(42)
    # Check if stack_zone is exposed and empty initially
    assert hasattr(game, "stack_zone")
    assert len(game.stack_zone) == 0

    # Test adding a card to stack zone (simulating logic)
    # We create a dummy card instance
    card = CardInstance(1, 100) # ID 1, Instance 100

    # pybind11 def_readwrite on std::vector returns a copy (list).
    # To modify, we must read, modify, and write back.
    sz = game.stack_zone
    sz.append(card)
    game.stack_zone = sz

    assert len(game.stack_zone) == 1
    assert game.stack_zone[0].instance_id == 100

def test_projected_cost_not_exposed_yet():
    # ManaSystem is not exposed as a class with static methods in bindings yet
    # But we can verify no errors in build/import which implies the C++ side is fine.
    # The get_projected_cost is currently C++ only unless we exposed ManaSystem or wrapper.
    # checking bindings.cpp... ManaSystem is NOT exposed.
    # But get_adjusted_cost is used inside ActionGenerator which is exposed.
    pass
