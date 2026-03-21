from dm_toolkit.gui.editor.validators_shared import ConditionValidator
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_static_allows_cards_matching_filter_existence_check():
    condition = {
        "type": "CARDS_MATCHING_FILTER",
        "op": ">=",
        "value": 1,
        "filter": {
            "owner": "SELF",
            "zones": ["BATTLE_ZONE"],
            "races": ["DRAGON"]
        }
    }

    errors = ConditionValidator.validate_static(condition)
    assert errors == [], f"Expected no validation errors, got: {errors}"


def test_static_allows_compare_stat_with_known_key():
    # Use a known key from CardTextResources
    stat_key = CardTextResources.COMPARE_STAT_EDITOR_KEYS[0]
    condition = {
        "type": "COMPARE_STAT",
        "stat_key": stat_key,
        "op": ">=",
        "value": 1
    }
    errors = ConditionValidator.validate_static(condition)
    assert errors == [], f"Expected no validation errors for COMPARE_STAT, got: {errors}"
