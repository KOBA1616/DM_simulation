# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QFormLayout, QComboBox, QWidget
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.schema_def import CommandSchema, FieldSchema, FieldType, get_schema, register_schema
from dm_toolkit.gui.editor.widget_factory import FormBuilder
from dm_toolkit.gui.localization import tr

class DynamicCommandForm(BaseEditForm):
    """
    A refactored Command Form that uses Schema Definitions to generate its UI.
    This serves as a proof-of-concept for the 'Improvement Measures'.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_schema = None
        self.dynamic_widgets = [] # Keep track to remove them on type change

        self.setup_static_ui()
        self.ensure_default_schemas() # Ensure schemas are registered (idempotent)

    def setup_static_ui(self):
        """Sets up the static parts of the form (Type selector)."""
        self.layout = QFormLayout(self)

        self.type_combo = QComboBox()
        self.register_widget(self.type_combo, 'type')
        self.layout.addRow(tr("Command Type"), self.type_combo)

        self.type_combo.currentIndexChanged.connect(self.on_type_changed)

    def ensure_default_schemas(self):
        """Define some schemas for testing/demo purposes if not already registered."""
        if get_schema("DRAW_CARD") is None:
            # Example: DRAW_CARD
            draw_schema = CommandSchema("DRAW_CARD", [
                FieldSchema("target_group", tr("Target"), FieldType.PLAYER, default="PLAYER_SELF"),
                FieldSchema("amount", tr("Cards to Draw"), FieldType.INT, default=1, min_value=1),
                FieldSchema("optional", tr("Optional"), FieldType.BOOL, default=False),
                FieldSchema("links", tr("Variable Links"), FieldType.LINK, produces_output=True)
            ])
            register_schema(draw_schema)

        if get_schema("DISCARD") is None:
            # Example: DISCARD
            discard_schema = CommandSchema("DISCARD", [
                FieldSchema("target_group", tr("Target"), FieldType.PLAYER, default="PLAYER_SELF"),
                FieldSchema("target_filter", tr("Filter"), FieldType.FILTER),
                FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
                FieldSchema("links", tr("Variable Links"), FieldType.LINK, produces_output=True)
            ])
            register_schema(discard_schema)

        # Populate combo
        self.populate_combo(self.type_combo, ["DRAW_CARD", "DISCARD"])

    def on_type_changed(self):
        type_name = self.type_combo.currentText() # simple for now
        self.rebuild_form(type_name)
        self.update_data() # Signal change

    def rebuild_form(self, type_name):
        """Rebuilds the dynamic part of the form based on schema."""
        # 1. Clear dynamic rows and delete widgets
        while self.layout.rowCount() > 1:
            # getLayoutItem(row, role) is not available directly on QFormLayout in older Qt versions easily
            # But takeAt(index) works for linear iteration, or removeRow(row)
            # removeRow deletes the layout item but not necessarily the widget if it has parent

            # Better approach: Iterate from end
            res = self.layout.takeRow(1) # RowResult
            if res.labelItem:
                w = res.labelItem.widget()
                if w: w.deleteLater()
            if res.fieldItem:
                w = res.fieldItem.widget()
                if w: w.deleteLater()

        # Clear bindings for dynamic widgets
        keys_to_remove = [k for k in self.bindings if k != 'type']
        for k in keys_to_remove:
            # Unregister widget from input_widgets set if present
            w = self.bindings[k]
            if w in self.input_widgets:
                self.input_widgets.remove(w)
            del self.bindings[k]

        # 2. Get Schema
        schema = get_schema(type_name)
        if not schema:
            return

        # 3. Build
        builder = FormBuilder(self)
        builder.build(schema, self.layout)

        self.active_schema = schema

    def _load_ui_from_data(self, data, item):
        """Load data using schema-aware logic."""
        # 1. Set type (triggers rebuild)
        t = data.get('type', 'DRAW_CARD') # Default

        # Use suppress_signals to prevent rebuild from triggering unintended updates during load
        with self.suppress_signals():
            self.set_combo_by_data(self.type_combo, t)
            # Note: set_combo_by_data triggers currentIndexChanged if not blocked.
            # BaseEditForm.suppress_signals blocks registered input widgets.
            # self.type_combo is registered.
            # BUT: on_type_changed is connected to currentIndexChanged.
            # If signals are blocked, on_type_changed won't fire.
            # So we must manually call rebuild_form if we block signals.
            self.rebuild_form(t)

            # 2. Apply bindings
            self._apply_bindings(data)

        # Handle Link Widget explicitly if key mismatch?
        # Schema uses 'links' key for widget, but data might be flat.
        # VariableLinkWidget usually handles flat data via get_data/set_data(whole_dict).
        # We might need to pass the whole data dict to the link widget.
        if 'links' in self.bindings:
            self.bindings['links'].set_data(data)

    def _save_ui_to_data(self, data):
        """Save data."""
        data['type'] = self.type_combo.currentText()

        self._collect_bindings(data)

        # Link widget special handling (it writes in-place usually)
        if 'links' in self.bindings:
            self.bindings['links'].get_data(data)
