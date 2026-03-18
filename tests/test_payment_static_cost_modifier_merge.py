from dm_toolkit import payment


def test_apply_passive_reductions_merges_cost_reductions_and_static_modifier() -> None:
    """PASSIVE and static COST_MODIFIER should be applied together conservatively."""
    card = {
        "cost_reductions": [
            {"id": "p1", "type": "PASSIVE", "amount": 2, "min_mana_cost": 3},
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value": 1},
        ],
    }

    adjusted = payment.apply_passive_reductions(card, base_cost=10, units=1)

    # 10 - (2 + 1) = 7; floor=3 does not clamp here.
    assert adjusted == 7


def test_evaluate_cost_includes_static_modifier_in_passive_stage() -> None:
    """evaluate_cost should include converted COST_MODIFIER in passive stage and active stage after that."""
    card = {
        "cost_reductions": [
            {"id": "p1", "type": "PASSIVE", "amount": 1},
            {"id": "a1", "type": "ACTIVE_PAYMENT", "reduction_per_unit": 2, "max_units": 1},
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value": 2},
        ],
    }

    plan = payment.evaluate_cost(
        card,
        base_cost=10,
        units=1,
        active_reduction_id="a1",
        active_units=1,
    )

    # passive stage: 10 - (1 + 2) = 7
    assert plan.adjusted_after_passive == 7
    assert plan.total_passive_reduction == 3
    # active stage: 7 - 2 = 5
    assert plan.final_cost == 5
