from dm_toolkit.gui.editor.validators_shared import ModifierValidator


def test_stat_scaled_payload_without_legacy_value_and_none_max_is_valid() -> None:
    """再発防止: Form保存後のSTAT_SCALED payloadがvalidatorを通ること。"""
    payload = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "CARDS_DRAWN_THIS_TURN",
        "per_value": 1,
        "min_stat": 0,
        "scope": "SELF",
    }

    errors = ModifierValidator.validate(payload)
    assert errors == [], f"Expected no validation errors, got: {errors}"
