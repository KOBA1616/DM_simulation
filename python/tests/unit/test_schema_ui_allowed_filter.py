from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
from dm_toolkit.gui.editor.schema_def import SchemaLoader, get_schema, SCHEMA_REGISTRY
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.gui.editor.forms.command_strategies import CommandUIStrategy


def test_schema_allowed_fields_fallback(monkeypatch):
    # Prepare a minimal config cache that includes allowed_filter_fields
    monkeypatch.setattr(EditorConfigLoader, '_config_cache', {
        'MEKRAID': {'visible': ['target_filter'], 'allowed_filter_fields': ['civilizations', 'races']}
    })

    # Ensure fresh registry
    SCHEMA_REGISTRY.clear()
    SchemaLoader.load_schemas()

    schema = get_schema('MEKRAID')
    assert schema is not None
    assert getattr(schema, 'allowed_filter_fields') == ['civilizations', 'races']

    # Now simulate the form path where COMMAND_UI_CONFIG doesn't provide allowed fields
    COMMAND_UI_CONFIG['MEKRAID'] = {'target_filter_visible': True}

    class DummyFilterWidget:
        def __init__(self):
            self.allowed = None

        def set_allowed_fields(self, v):
            self.allowed = v

        def setTitle(self, t):
            pass

    class DummyForm:
        def __init__(self):
            self.filter_group = type('G', (), {'setVisible': lambda self, v: None})()
            self.filter_widget = DummyFilterWidget()
            # Minimal placeholders for attributes referenced by update_visibility
            placeholder = type('W', (), {'setVisible': lambda self, v: None, 'setText': lambda self, t: None})
            attrs = (
                'val1_label', 'val1_spin', 'val2_label', 'val2_spin', 'str_label', 'str_edit',
                'mutation_kind_label', 'mutation_kind_container', 'source_zone_label', 'source_zone_combo',
                'dest_zone_label', 'dest_zone_combo', 'measure_mode_combo', 'stat_key_combo', 'stat_key_label',
                'stat_preset_btn', 'ref_mode_combo', 'option_count_label', 'option_count_spin', 'generate_options_btn',
                'select_count_label', 'select_count_spin', 'mutation_kind_combo', 'mutation_kind_edit', 'allow_duplicates_label',
                'allow_duplicates_check', 'up_to_check'
            )
            for name in attrs:
                setattr(self, name, placeholder())

    form = DummyForm()
    strategy = CommandUIStrategy('MEKRAID')
    strategy.update_visibility(form)

    # Expect the UI to receive allowed fields from the schema as fallback
    assert form.filter_widget.allowed == ['civilizations', 'races']
