from dm_toolkit.gui.editor.forms.unified_action_form import (
    _resolve_query_mode_from_params,
    _sync_query_mode_params,
)


def test_legacy_empty_query_mode_can_be_normalized_to_default() -> None:
    params = {"str_param": "", "query_mode": "", "query_string": ""}
    mode = _resolve_query_mode_from_params(params)
    assert mode == ""

    # mimic legacy-empty normalization branch used by save flow
    if any(k in params for k in ("str_param", "query_mode", "query_string")) and not mode:
        mode = "CARDS_MATCHING_FILTER"
    _sync_query_mode_params(params, mode)

    assert params["str_param"] == "CARDS_MATCHING_FILTER"
    assert params["query_mode"] == "CARDS_MATCHING_FILTER"
    assert params["query_string"] == "CARDS_MATCHING_FILTER"
