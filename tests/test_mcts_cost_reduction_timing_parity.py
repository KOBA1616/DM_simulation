import json
import pytest

dm = pytest.importorskip("dm_ai_module")


def test_mcts_oneshot_triggers_recalc_and_matches_normal_path() -> None:
    """
    RED test: Ensure resolve_command_oneshot path triggers ContinuousEffectSystem::recalculate
    so STAT_SCALED modifiers are up-to-date (parity with normal path).
    """
    # Minimal card DB: playable card and a provider with STAT_SCALED cost modifier
    cards = [
        {"id": 100, "name": "Playable", "cost": 2},
        {
            "id": 200,
            "name": "Provider",
            "cost": 0,
            "static_abilities": [
                {
                    "type": "COST_MODIFIER",
                    "value_mode": "STAT_SCALED",
                    "stat_key": "CREATURES_PLAYED",
                    "per_value": 1,
                    "min_stat": 1,
                    "max_reduction": 3,
                }
            ],
        },
    ]

    db = dm.JsonLoader.load_cards(json.dumps(cards))
    game = dm.GameInstance(0, db)
    state = game.state

    # Place provider into battle (instance id 201) and playable into hand (101)
    try:
        state.add_test_card_to_battle(0, 200, 201, False, False)
    except TypeError:
        # fallback shapes
        state.add_test_card_to_battle(0, 200, 201)

    try:
        state.add_card_to_hand(0, 100, 101)
    except TypeError:
        state.add_card_to_hand(0, 100)

    # Normal path: set stat to 0 -> no reduction
    game.execute_command(dm.StatCommand(dm.StatType.CREATURES_PLAYED, 0))
    reduction_zero = sum(getattr(m, 'reduction_amount', 0) for m in getattr(state, 'active_modifiers', []))

    # Update stat via normal path to 2 -> recalc should update modifiers
    game.execute_command(dm.StatCommand(dm.StatType.CREATURES_PLAYED, 2))
    reduction_after_stat = sum(getattr(m, 'reduction_amount', 0) for m in getattr(state, 'active_modifiers', []))
    assert reduction_after_stat > reduction_zero

    # Now call one-shot path via PipelineExecutor.execute_command (MANA_CHARGE route)
    # This binding is only available in native builds; skip in Python-only fallback.
    if not hasattr(dm, 'PipelineExecutor'):
        pytest.skip("PipelineExecutor/resolve_command_oneshot binding not available in this build")

    exec = dm.PipelineExecutor()
    # Invoke MANA_CHARGE oneshot (binding routes this through resolve_command_oneshot)
    exec.execute_command({"type": "MANA_CHARGE", "instance_id": 201}, state)

    reduction_after_oneshot = sum(getattr(m, 'reduction_amount', 0) for m in getattr(state, 'active_modifiers', []))

    # Parity: oneshot recalc should observe same reduction as normal path
    assert reduction_after_oneshot == reduction_after_stat
