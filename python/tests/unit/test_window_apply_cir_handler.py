from dm_toolkit.gui.editor.window import CardEditor


class DummyUnifiedForm:
    def __init__(self):
        self.called_with = None

    def apply_cir(self, cir):
        self.called_with = cir
        return True


class DummyInspector:
    def __init__(self):
        self.unified_form = DummyUnifiedForm()


class DummyItem:
    def __init__(self, ident, data_type):
        self._id = ident
        self._data_type = data_type

    def index(self):
        return self._id

    def data(self, role=None):
        return self._data_type


def test_apply_cir_handler_calls_unified_form():
    editor = CardEditor.__new__(CardEditor)
    editor.inspector = DummyInspector()

    card_item = DummyItem('card_idx', 'CARD')
    item = DummyItem('cmd_idx', 'COMMAND')

    payload = {'cir': [{'type': 'TEST', 'payload': {'amount': 1}}]}
    handlers = editor._structure_handlers(card_item, item, 'COMMAND', payload)

    handler = handlers.get('APPLY_CIR')
    assert handler is not None

    res = handler()
    # Handler should return False (no tree mutation) but call apply_cir
    assert res is False
    assert editor.inspector.unified_form.called_with == payload['cir']
