from dm_toolkit.gui.editor.widget_factory import _format_select_option_label
from dm_toolkit.gui.editor.schema_def import FieldSchema, FieldType


def test_query_mode_special_labels_are_human_readable() -> None:
    schema = FieldSchema(
        "str_param",
        "Query Mode",
        FieldType.SELECT,
        options=["CARDS_MATCHING_FILTER", "SELECT_TARGET", "SELECT_OPTION"],
    )

    assert _format_select_option_label(schema, "CARDS_MATCHING_FILTER") != "CARDS_MATCHING_FILTER"
    assert _format_select_option_label(schema, "SELECT_TARGET") != "SELECT_TARGET"
    assert _format_select_option_label(schema, "SELECT_OPTION") != "SELECT_OPTION"


def test_query_mode_stat_key_uses_stat_label() -> None:
    schema = FieldSchema(
        "str_param",
        "Query Mode",
        FieldType.SELECT,
        options=["HAND_COUNT"],
    )

    label = _format_select_option_label(schema, "HAND_COUNT")
    assert label != "HAND_COUNT"
