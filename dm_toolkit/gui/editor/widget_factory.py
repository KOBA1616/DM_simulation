# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QCheckBox, QHBoxLayout, QComboBox, QLabel, QVBoxLayout, QPushButton
)
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_value_spin, make_measure_mode_combo, make_ref_mode_combo,
    make_scope_combo, make_option_controls
)
from dm_toolkit.consts import GRANTABLE_KEYWORDS

class WidgetFactory:
    """Factory class to create widgets based on schema definitions."""

    @staticmethod
    def create_widget(parent, field_config, update_callback):
        """
        Creates a widget based on the field configuration.

        Args:
            parent: The parent widget (usually the form).
            field_config (dict): The configuration for the field.
            update_callback (callable): The function to call when data changes.

        Returns:
            QWidget: The created widget.
        """
        w_type = field_config.get('widget', 'text')
        label = field_config.get('label')

        widget = None

        if w_type == 'text':
            widget = QLineEdit()
            widget.textChanged.connect(update_callback)

        elif w_type == 'spinbox':
            widget = make_value_spin(parent)
            widget.valueChanged.connect(update_callback)
            if 'default' in field_config:
                widget.setValue(field_config['default'])

        elif w_type == 'checkbox':
            widget = QCheckBox(tr(label) if label else "")
            # Label is integrated into checkbox usually, but schema might provide it separately
            widget.stateChanged.connect(update_callback)

        elif w_type == 'player_scope':
            widget = WidgetFactory._create_player_scope(update_callback)

        elif w_type == 'zone_combo':
            widget = QComboBox()
            zones = ["NONE", "HAND", "BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "SHIELD_ZONE", "DECK"]
            for z in zones:
                widget.addItem(tr(z), z)
            widget.currentIndexChanged.connect(update_callback)

        elif w_type == 'scope_combo':
            widget = make_scope_combo(parent, include_zones=True)
            widget.currentIndexChanged.connect(update_callback)

        elif w_type == 'query_mode_combo':
            widget = make_measure_mode_combo(parent)
            widget.currentIndexChanged.connect(update_callback)

        elif w_type == 'ref_mode_combo':
            widget = make_ref_mode_combo(parent)
            widget.currentIndexChanged.connect(update_callback)

        elif w_type == 'keyword_combo':
            widget = QComboBox()
            for kw in GRANTABLE_KEYWORDS:
                widget.addItem(tr(kw), kw)
            widget.currentIndexChanged.connect(update_callback)

        elif w_type == 'filter_editor':
            widget = FilterEditorWidget()
            widget.filterChanged.connect(update_callback)
            if 'title' in field_config:
                widget.setTitle(tr(field_config['title']))
            if 'allowed_fields' in field_config:
                widget.set_allowed_fields(field_config['allowed_fields'])

        elif w_type == 'variable_link':
            widget = VariableLinkWidget()
            widget.linkChanged.connect(update_callback)
            if field_config.get('produces_output'):
                if hasattr(widget, 'set_output_hint'):
                    widget.set_output_hint(True)
            if 'output_label' in field_config:
                widget.output_label_text = tr(field_config['output_label'])

        elif w_type == 'options_control':
            widget = WidgetFactory._create_options_control(parent, update_callback)

        return widget

    @staticmethod
    def _create_player_scope(callback):
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0,0,0,0)

        self_chk = QCheckBox(tr("Self"))
        opp_chk = QCheckBox(tr("Opponent"))

        def check_state(state):
            if not self_chk.isChecked() and not opp_chk.isChecked():
                 self_chk.setChecked(True)
            callback()

        self_chk.stateChanged.connect(check_state)
        opp_chk.stateChanged.connect(check_state)

        h_layout.addWidget(self_chk)
        h_layout.addWidget(opp_chk)

        # Attach sub-widgets for easy access
        container.self_chk = self_chk
        container.opp_chk = opp_chk

        return container

    @staticmethod
    def _create_options_control(parent, callback):
        # This requires the parent to handle request_generate_options logic usually,
        # but here we encapsulate as much as possible.
        spin, btn, lbl, layout = make_option_controls(parent)

        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0,0,0,0)

        top_row = QWidget()
        h_layout = QHBoxLayout(top_row)
        h_layout.setContentsMargins(0,0,0,0)
        h_layout.addWidget(lbl)
        h_layout.addWidget(spin)
        h_layout.addWidget(btn)

        v_layout.addWidget(top_row)
        v_layout.addLayout(layout)

        container.spin = spin
        container.btn = btn
        container.option_layout = layout

        # We need to bridge the button click to the parent's handler if possible,
        # or expose the signal.
        if hasattr(parent, 'request_generate_options'):
             btn.clicked.connect(parent.request_generate_options)

        # Also connect spin change to update data
        spin.valueChanged.connect(callback)

        return container
