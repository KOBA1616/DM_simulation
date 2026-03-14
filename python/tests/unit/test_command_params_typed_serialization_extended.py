from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor.models import QueryParams, TransitionParams, ModifierParams


def test_query_params_ingest_and_serialize():
    raw = {"type": "QUERY", "params": {"query_string": "find", "options": ["A", "B"]}}
    cmd = CommandModel.model_validate(raw)
    assert isinstance(cmd.params, QueryParams)
    out = cmd.model_dump()
    assert out.get("query_string") == "find"
    assert out.get("options") == ["A", "B"]


def test_transition_params_ingest_and_serialize():
    raw = {"type": "TRANSITION", "params": {"target_state": "NEXT", "reason": "auto"}}
    cmd = CommandModel.model_validate(raw)
    assert isinstance(cmd.params, TransitionParams)
    out = cmd.model_dump()
    assert out.get("target_state") == "NEXT"
    assert out.get("reason") == "auto"


def test_modifier_params_ingest_and_serialize():
    raw = {"type": "MODIFY", "params": {"amount": 5, "scope": "SELF"}}
    cmd = CommandModel.model_validate(raw)
    assert isinstance(cmd.params, ModifierParams)
    out = cmd.model_dump()
    assert out.get("amount") == 5
    assert out.get("scope") == "SELF"
