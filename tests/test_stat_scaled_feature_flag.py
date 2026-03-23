import pytest

from dm_toolkit.payment import evaluate_cost


def test_stat_scaled_disabled_by_feature_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    card = {
        'static_abilities': [
            {
                'type': 'COST_MODIFIER',
                'value_mode': 'STAT_SCALED',
                'stat_key': 'summon_count',
                'per_value': 1,
                'min_stat': 1,
            }
        ]
    }
    base_cost = 5
    stat_values = {'summon_count': 3}

    monkeypatch.setenv('STAT_SCALED_ENABLED', '0')
    plan = evaluate_cost(card, base_cost, units=1, stat_values=stat_values)

    # rollback mode: STAT_SCALED is ignored, so no reduction is applied
    assert plan.final_cost == 5


def test_stat_scaled_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    card = {
        'static_abilities': [
            {
                'type': 'COST_MODIFIER',
                'value_mode': 'STAT_SCALED',
                'stat_key': 'summon_count',
                'per_value': 1,
                'min_stat': 1,
            }
        ]
    }

    monkeypatch.delenv('STAT_SCALED_ENABLED', raising=False)
    plan = evaluate_cost(card, 5, units=1, stat_values={'summon_count': 3})

    assert plan.final_cost == 2
