from dm_toolkit.gui.editor.validators_shared import detect_passive_static_conflicts


def test_detect_passive_static_conflicts_warns_when_both_sources_exist() -> None:
    card = {
        "cost_reductions": [
            {"id": "p1", "type": "PASSIVE", "reduction_amount": 1}
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value": 1}
        ],
    }

    warnings = detect_passive_static_conflicts(card)
    assert len(warnings) == 1
    assert "PASSIVE" in warnings[0]
    assert "COST_MODIFIER" in warnings[0]


def test_detect_passive_static_conflicts_no_warning_with_single_source() -> None:
    card = {
        "cost_reductions": [
            {"id": "p1", "type": "PASSIVE", "reduction_amount": 1}
        ],
        "static_abilities": [],
    }

    warnings = detect_passive_static_conflicts(card)
    assert warnings == []
