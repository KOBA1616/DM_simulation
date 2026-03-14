class DummyWidget:
    def __init__(self):
        self._value = None
    def set_value(self, v):
        self._value = v


def test_apply_cir_updates_widgets_and_model():
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    from dm_toolkit.gui.editor.models import CommandModel

    form = UnifiedActionForm()
    # prepare current model and widgets_map
    form.current_model = CommandModel.construct(type='TEST', params={})
    w_amount = DummyWidget()
    w_name = DummyWidget()
    form.widgets_map['amount'] = w_amount
    form.widgets_map['str_val'] = w_name

    cir = [{'type': 'TEST', 'payload': {'amount': 5, 'str_val': 'X'}}]

    updated = form.apply_cir(cir)

    assert updated is True
    assert form.current_model.params['amount'] == 5
    assert form.current_model.params['str_val'] == 'X'
    assert w_amount._value == 5
    assert w_name._value == 'X'
