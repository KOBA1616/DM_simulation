# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


def test_load_ui_accepts_typed_params_after_command_ingest_without_crashing():
    form = UnifiedActionForm(None)

    data = {"type": "DRAW_CARD", "amount": 2}

    form._load_ui_from_data(data, None)

    amount_widget = form.widgets_map.get("amount")
    assert amount_widget is not None
    assert hasattr(amount_widget, "get_value")
    assert amount_widget.get_value() == 2


class _StubValueWidget:
    def __init__(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def setStyleSheet(self, _style):
        return None

    def setToolTip(self, _tooltip):
        return None


def test_save_ui_handles_typed_params_without_item_assignment_error():
    form = UnifiedActionForm(None)
    idx = form.type_combo.findData("MUTATE")
    if idx < 0:
        form.type_combo.addItem("MUTATE", "MUTATE")
        idx = form.type_combo.findData("MUTATE")
    form.type_combo.setCurrentIndex(idx)

    # mutation_kind is stored in params for MUTATE and previously crashed on typed params assignment.
    form.widgets_map = {
        "mutation_kind": _StubValueWidget("GRANT_ABILITY"),
        "amount": _StubValueWidget(1000),
        "target_filter": _StubValueWidget({"types": ["CREATURE"]}),
    }

    out = {}
    form._save_ui_to_data(out)

    assert out.get("type") == "MUTATE"
    # Mutate params may be flattened by serializer; accept either shape.
    mutation_kind = out.get("mutation_kind") or out.get("params", {}).get("mutation_kind")
    assert mutation_kind in ("GRANT_ABILITY", "MutationKind.GRANT_ABILITY")
