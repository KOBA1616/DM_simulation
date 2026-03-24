from dm_toolkit.gui.editor.models import ModifierModel


def test_modifier_model_keeps_stat_scaled_fields() -> None:
    m = ModifierModel(
        type="COST_MODIFIER",
        value_mode="STAT_SCALED",
        stat_key="CARDS_DRAWN_THIS_TURN",
        per_value=1,
        min_stat=1,
        max_reduction=4,
        increment_cost=1,
        scope="ALL",
    )

    dumped = m.model_dump(exclude_none=True)

    assert dumped["value_mode"] == "STAT_SCALED"
    assert dumped["stat_key"] == "CARDS_DRAWN_THIS_TURN"
    assert dumped["per_value"] == 1
    assert dumped["min_stat"] == 1
    assert dumped["max_reduction"] == 4
    assert dumped["increment_cost"] == 1


def test_modifier_model_allows_forward_compatible_extra_fields() -> None:
    m = ModifierModel(type="COST_MODIFIER", value_mode="STAT_SCALED", custom_extra=123)
    dumped = m.model_dump(exclude_none=True)

    assert dumped.get("custom_extra") == 123
