from dm_toolkit.gui.editor.forms.unified_action_form import (
    _resolve_query_mode_from_params,
    _sync_query_mode_params,
)


def test_resolve_query_mode_prefers_str_param_then_legacy() -> None:
    assert _resolve_query_mode_from_params({"str_param": "A", "query_mode": "B"}) == "A"
    assert _resolve_query_mode_from_params({"query_mode": "B", "query_string": "C"}) == "B"
    assert _resolve_query_mode_from_params({"query_string": "C"}) == "C"


def test_sync_query_mode_updates_all_compat_keys() -> None:
    params = {}
    _sync_query_mode_params(params, "SELECT_TARGET")

    assert params["str_param"] == "SELECT_TARGET"
    assert params["query_mode"] == "SELECT_TARGET"
    assert params["query_string"] == "SELECT_TARGET"
