from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


def test_infer_integrity_warning_field_keys_query_mode_only() -> None:
    form = UnifiedActionForm(None)
    warns = ["未設定: QUERY[0](QUERY) に Query Mode (str_param) が設定されていません"]

    keys = form._infer_integrity_warning_field_keys(warns)

    assert keys == {"str_param"}


def test_infer_integrity_warning_field_keys_select_option_amount() -> None:
    form = UnifiedActionForm(None)
    warns = [
        "未設定: QUERY[0](QUERY) SELECT_OPTION の選択数 (amount または input_value_key) が設定されていません"
    ]

    keys = form._infer_integrity_warning_field_keys(warns)

    assert "amount" in keys
    assert "input_value_key" in keys
