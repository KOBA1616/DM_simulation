# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QFormLayout, QComboBox, QWidget
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.schema_def import CommandSchema, FieldSchema, FieldType, get_schema, register_schema
from dm_toolkit.gui.editor.widget_factory import FormBuilder
from dm_toolkit.gui.editor.schema_config import register_all_schemas
from dm_toolkit.gui.i18n import tr

class DynamicCommandForm(BaseEditForm):
    """
    A refactored Command Form that uses Schema Definitions to generate its UI.
    This serves as a proof-of-concept for the 'Improvement Measures'.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_schema = None

        self.setup_static_ui()
        self.ensure_schemas()

    def setup_static_ui(self):
        """Sets up the static parts of the form (Type selector)."""
        self.layout = QFormLayout(self)

        self.type_combo = QComboBox()
        self.register_widget(self.type_combo, 'type')
        self.layout.addRow(tr("Command Type"), self.type_combo)

        self.type_combo.currentIndexChanged.connect(self.on_type_changed)

    def ensure_schemas(self):
        """Registers all schemas and populates the type combo."""
        register_all_schemas()

        # Populate combo from registry, sorted
        from dm_toolkit.gui.editor.schema_def import SCHEMA_REGISTRY
        types = sorted(list(SCHEMA_REGISTRY.keys()))
        self.populate_combo(self.type_combo, types)

    def on_type_changed(self):
        type_name = self.type_combo.currentText()
        self.rebuild_form(type_name)
        self.update_data() # Signal change

    def rebuild_form(self, type_name):
        """Rebuilds the dynamic part of the form based on schema."""
        # 1. Clear dynamic rows (everything after row 0)
        while self.layout.rowCount() > 1:
            res = self.layout.takeRow(1) # Remove row at index 1 repeatedly
            if res.labelItem:
                w = res.labelItem.widget()
                if w: w.deleteLater()
            if res.fieldItem:
                w = res.fieldItem.widget()
                if w: w.deleteLater()

        # 2. Clear bindings for dynamic widgets
        # We need to preserve 'type' binding
        keys_to_remove = [k for k in self.bindings if k != 'type']
        for k in keys_to_remove:
            w = self.bindings[k]
            # Safely unregister
            if w in self.input_widgets:
                self.input_widgets.remove(w)
            del self.bindings[k]

        # 3. Get Schema
        schema = get_schema(type_name)
        if not schema:
            return

        # 4. Build
        builder = FormBuilder(self)
        builder.build(schema, self.layout)

        self.active_schema = schema

    def _load_ui_from_data(self, data, item):
        """Load data using schema-aware logic."""
        # 1. Set type (triggers rebuild)
        t = data.get('type', 'NONE')

        # Use suppress_signals to prevent unintended updates during load
        # However, we must ensure rebuild_form is called.
        with self.suppress_signals():
            self.set_combo_by_data(self.type_combo, t)
            # Since suppress_signals blocks widget signals, the currentIndexChanged won't fire on type_combo.
            # We must manually trigger the rebuild logic.
            self.rebuild_form(t)

            # 2. Apply bindings
            self._apply_bindings(data)

            # Special handling for VariableLinkWidget if schema uses 'links'
            if 'links' in self.bindings:
                self.bindings['links'].set_data(data)

    def _save_ui_to_data(self, data):
        """Save data."""
        data['type'] = self.type_combo.currentText()

        # Save standard bindings
        self._collect_bindings(data)

        # Link widget special handling
        if 'links' in self.bindings:
            self.bindings['links'].get_data(data)
