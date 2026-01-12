# -*- coding: cp932 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QScrollArea, QPushButton, QMenu, QGroupBox, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.consts import (
    STRUCT_CMD_ADD_CHILD_EFFECT, STRUCT_CMD_ADD_SPELL_SIDE, STRUCT_CMD_REMOVE_SPELL_SIDE
)
from dm_toolkit.gui.editor.configs.card_config import CARD_SCHEMA
from dm_toolkit.gui.editor.widget_factory import WidgetFactory
from dm_toolkit.gui.editor.schema_def import FieldSchema

class CardEditForm(BaseEditForm):
    # Signal to request structural changes in the Logic Tree
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets_map = {} # Key -> Widget
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.form_layout = QFormLayout(self.scroll_content)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        # AI Configuration Section Container
        self.ai_group = QGroupBox(tr("AI Configuration"))
        self.ai_layout = QFormLayout(self.ai_group)
        self.ai_widgets_created = False

        # Generate Fields from Schema
        for field in CARD_SCHEMA.fields:
            self._create_field_widget(field)

        # Add AI Group at the end if widgets were added to it
        if self.ai_widgets_created:
            self.form_layout.addRow(self.ai_group)

        # Actions Section
        actions_group = QGroupBox(tr("Actions"))
        actions_layout = QVBoxLayout(actions_group)

        self.add_effect_btn = QPushButton(tr("Add Effect"))
        self.add_effect_btn.clicked.connect(self.on_add_effect_clicked)
        actions_layout.addWidget(self.add_effect_btn)

        self.form_layout.addRow(actions_group)

    def _create_field_widget(self, field: FieldSchema):
        # Determine target layout
        layout = self.form_layout
        if field.key in ['is_key_card', 'ai_importance_score']:
            layout = self.ai_layout
            self.ai_widgets_created = True

        # Special callback wrapper to handle structural side-effects
        def update_wrapper():
            self.update_data()
            self._check_side_effects(field.key)

        widget = WidgetFactory.create_widget(self, field, update_wrapper)
        if widget:
            self.widgets_map[field.key] = widget

            # Special handling for ID (hidden)
            if field.key == 'id':
                widget.setVisible(False)
                # Don't add label for hidden ID
                return

            # Add to layout
            # For boolean/checkbox, the widget might already contain the label or be self-labeled
            if field.field_type.name == 'BOOL':
                 layout.addRow(widget)
            else:
                 layout.addRow(tr(field.label), widget)

    def _check_side_effects(self, key):
        """Handle structural changes based on field updates."""
        if key == 'twinpact':
            widget = self.widgets_map.get(key)
            if widget:
                is_checked = widget.isChecked()
                if is_checked:
                    self.structure_update_requested.emit(STRUCT_CMD_ADD_SPELL_SIDE, {})
                else:
                    self.structure_update_requested.emit(STRUCT_CMD_REMOVE_SPELL_SIDE, {})

        if key == 'type':
            self._update_visibility()

    def _update_visibility(self):
        """Update field visibility based on dependencies defined in schema."""
        # Get current values
        current_values = {}
        for k, w in self.widgets_map.items():
            if hasattr(w, 'get_value'):
                current_values[k] = w.get_value()
            elif hasattr(w, 'currentText'):
                current_values[k] = w.currentText()

        for field in CARD_SCHEMA.fields:
            if field.visible_if:
                widget = self.widgets_map.get(field.key)
                if not widget: continue

                visible = True
                for dep_key, allowed_vals in field.visible_if.items():
                    curr = current_values.get(dep_key)
                    if isinstance(allowed_vals, list):
                        if curr not in allowed_vals:
                            visible = False
                            break
                    else:
                        if curr != allowed_vals:
                            visible = False
                            break

                widget.setVisible(visible)
                # Find label? Layout management makes this tricky without referencing the label item.
                # QFormLayout.itemAt(row, role) -> but we don't track rows easily.
                # Simple workaround: The user won't see the widget, the label remains.
                # Better: WidgetFactory should ideally return (label, widget) or we track labels.

                # For now, if the widget is hidden, the row might look empty.
                # This is a limitation of this quick refactor.
                # To fix, we'd need to store the layout row or the label widget.

                # Iterate layout to find label associated with widget
                # This is expensive but correct
                label = self.form_layout.labelForField(widget)
                if label:
                    label.setVisible(visible)

    def on_add_effect_clicked(self):
        menu = QMenu(self)

        kw_act = menu.addAction(tr("Keyword Ability"))
        if kw_act is not None:
            kw_act.triggered.connect(lambda: self.structure_update_requested.emit(STRUCT_CMD_ADD_CHILD_EFFECT, {"type": "KEYWORDS"}))

        trig_act = menu.addAction(tr("Triggered Ability"))
        if trig_act is not None:
            trig_act.triggered.connect(lambda: self.structure_update_requested.emit(STRUCT_CMD_ADD_CHILD_EFFECT, {"type": "TRIGGERED"}))

        static_act = menu.addAction(tr("Static Ability"))
        if static_act is not None:
            static_act.triggered.connect(lambda: self.structure_update_requested.emit(STRUCT_CMD_ADD_CHILD_EFFECT, {"type": "STATIC"}))

        react_act = menu.addAction(tr("Reaction Ability"))
        if react_act is not None:
            react_act.triggered.connect(lambda: self.structure_update_requested.emit(STRUCT_CMD_ADD_CHILD_EFFECT, {"type": "REACTION"}))

        menu.exec(QCursor.pos())

    def _load_ui_from_data(self, data, item):
        if not data: data = {}

        # Pre-process data to match flat schema (e.g. keywords extraction)
        flat_data = data.copy()

        # Flatten keywords
        keywords = data.get('keywords', {})
        if keywords.get('hyper_energy'): flat_data['hyper_energy'] = True

        # Structural check for Twinpact
        has_spell_side = False
        if item:
            for i in range(item.rowCount()):
                child = item.child(i)
                if child is None: continue
                child_type = child.data(Qt.ItemDataRole.UserRole + 1)
                if child_type == "SPELL_SIDE":
                    has_spell_side = True
                    break
        flat_data['twinpact'] = has_spell_side

        # Set values to widgets
        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'set_value'):
                val = flat_data.get(key)

                # Default handling
                if val is None:
                    # Look up default in schema
                    field = next((f for f in CARD_SCHEMA.fields if f.key == key), None)
                    if field and field.default is not None:
                         val = field.default

                    # Type-specific empty defaults
                    if field and field.field_type.name == 'RACES': val = []
                    if field and field.field_type.name == 'CIVILIZATION': val = []

                if val is not None:
                    # Block signals during load
                    old_state = widget.blockSignals(True)
                    widget.set_value(val)
                    widget.blockSignals(old_state)

        self._update_visibility()

    def _save_ui_to_data(self, data):
        """
        Save UI values back into data dict.
        """
        new_data = data.copy()

        # Collect from widgets
        for key, widget in self.widgets_map.items():
            if hasattr(widget, 'get_value'):
                new_data[key] = widget.get_value()

        # Post-process (Deep structure reconstruction)

        # Keywords reconstruction
        current_keywords = new_data.get('keywords', {})
        if not isinstance(current_keywords, dict): current_keywords = {}

        # Handle Evolution/Neo flags based on Type
        type_str = new_data.get('type', 'CREATURE')
        if 'evolution' in current_keywords: del current_keywords['evolution']
        if 'neo' in current_keywords: del current_keywords['neo']
        if 'g_neo' in current_keywords: del current_keywords['g_neo']

        if type_str == "EVOLUTION_CREATURE":
            current_keywords['evolution'] = True
        elif type_str == "NEO_CREATURE":
            current_keywords['evolution'] = True
            current_keywords['neo'] = True
        elif type_str == "G_NEO_CREATURE":
            current_keywords['evolution'] = True
            current_keywords['neo'] = True
            current_keywords['g_neo'] = True

        # Hyper Energy
        if new_data.get('hyper_energy'):
            current_keywords['hyper_energy'] = True
        elif 'hyper_energy' in current_keywords:
            del current_keywords['hyper_energy']

        # Remove temporary fields from data
        if 'hyper_energy' in new_data: del new_data['hyper_energy']
        if 'twinpact' in new_data: del new_data['twinpact']

        # Ensure spell power is 0
        if type_str == "SPELL":
            new_data['power'] = 0

        new_data['keywords'] = current_keywords

        # Apply back to data object
        data.clear()
        data.update(new_data)

    def _get_display_text(self, data):
        return f"{data.get('id', 0)} - {data.get('name', '')}"
