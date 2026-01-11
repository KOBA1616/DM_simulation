# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QComboBox, QLineEdit
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG
from dm_toolkit.consts import GRANTABLE_KEYWORDS

class CommandUIStrategy:
    """Base strategy for handling command-specific UI logic."""
    def __init__(self, command_type):
        self.command_type = command_type

    def get_config(self):
        return COMMAND_UI_CONFIG.get(self.command_type, {})

    def update_visibility(self, form):
        """Updates widget visibility and labels based on configuration."""
        cfg = self.get_config()

        # Standard Visibility Logic (moved from UnifiedActionForm)
        form.val1_label.setVisible(cfg.get('amount_visible', False))
        form.val1_spin.setVisible(cfg.get('amount_visible', False))
        form.up_to_check.setVisible('up_to' in cfg.get('visible', []))
        form.val1_label.setText(tr(cfg.get('amount_label', 'Amount')))

        form.val2_label.setVisible(cfg.get('val2_visible', False))
        form.val2_spin.setVisible(cfg.get('val2_visible', False))
        form.val2_label.setText(tr(cfg.get('val2_label', 'Value 2')))

        form.str_label.setVisible(cfg.get('str_param_visible', False))
        form.str_edit.setVisible(cfg.get('str_param_visible', False))
        if cfg.get('str_param_label'):
            form.str_label.setText(tr(cfg.get('str_param_label')))

        form.mutation_kind_label.setVisible(cfg.get('mutation_kind_visible', False))
        if form.mutation_kind_container:
            form.mutation_kind_container.setVisible(cfg.get('mutation_kind_visible', False))
        if cfg.get('mutation_kind_label'):
            form.mutation_kind_label.setText(tr(cfg.get('mutation_kind_label')))

        form.source_zone_label.setVisible(cfg.get('from_zone_visible', False))
        form.source_zone_combo.setVisible(cfg.get('from_zone_visible', False))
        form.dest_zone_label.setVisible(cfg.get('to_zone_visible', False))
        form.dest_zone_combo.setVisible(cfg.get('to_zone_visible', False))

        # Defaults for special widgets (hidden by default)
        form.measure_mode_combo.setVisible(False)
        form.stat_key_combo.setVisible(False)
        form.stat_key_label.setVisible(False)
        form.stat_preset_btn.setVisible(False)
        form.ref_mode_combo.setVisible(False)
        form.option_count_label.setVisible(False)
        form.option_count_spin.setVisible(False)
        form.generate_options_btn.setVisible(False)
        form.select_count_label.setVisible(False)
        form.select_count_spin.setVisible(False)

        # Filter config
        form.filter_group.setVisible(cfg.get('target_filter_visible', False))
        if cfg.get('target_filter_visible', False):
             try:
                 form.filter_widget.set_allowed_fields(cfg.get('allowed_filter_fields', None))
                 # Reset title to default unless overridden
                 form.filter_widget.setTitle(tr("Filter"))
             except Exception:
                 pass

    def load_data(self, form, data):
        """Maps data to widgets."""
        # Standard mapping is handled by BaseEditForm bindings usually,
        # but UnifiedForm does explicit mapping for some fields.
        # We rely on UnifiedForm's standard mapping for common fields,
        # and this method handles strategy-specific overrides.
        pass

    def save_data(self, form, data):
        """Maps widgets to data."""
        pass


class DefaultStrategy(CommandUIStrategy):
    pass


class QueryStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        form.measure_mode_combo.setVisible(True)
        form.stat_key_combo.setVisible(True)
        form.stat_key_label.setVisible(True)
        form.stat_preset_btn.setVisible(True)

    def load_data(self, form, data):
        stat_key = data.get('str_param', '')
        if stat_key and hasattr(form, 'stat_key_combo'):
            form.set_combo_by_data(form.stat_key_combo, stat_key)


class MutateStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        # Check current mutation kind to see if we need ref mode
        mk = form.mutation_kind_combo.currentData() if form.mutation_kind_combo.isVisible() else form.mutation_kind_edit.text()
        # In UnifiedForm, mutation kind combo is used if type is ADD_KEYWORD (not MUTATE usually),
        # but MUTATE might use text edit. UnifiedForm logic is a bit mixed.
        # Let's check the form's logic: ref_mode is shown if mutation_kind_combo.currentData() == 'COST_REFERENCE'.
        # But MUTATE usually uses line edit unless it's REVOLUTION_CHANGE.

        # Actually UnifiedForm logic:
        # self.ref_mode_combo.setVisible(t == 'MUTATE' and self.mutation_kind_combo.currentData() == 'COST_REFERENCE')
        # But mutation_kind_combo is only visible for ADD_KEYWORD?
        # Wait, setup_ui adds both to a stacked widget.

        if form.mutation_kind_combo.isVisible() and form.mutation_kind_combo.currentData() == 'COST_REFERENCE':
             form.ref_mode_combo.setVisible(True)

    def save_data(self, form, data):
        cfg = self.get_config()
        if cfg.get('mutation_kind_visible'):
            # This is handled in generic save, but let's ensure
            pass


class MekraidStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        form.val1_label.setText(tr('Max Cost'))
        form.val1_label.setVisible(True)
        form.val1_spin.setVisible(True)

        form.val2_label.setText(tr('Look Count'))
        form.val2_label.setVisible(True)
        form.val2_spin.setVisible(True)

        form.select_count_label.setText(tr('Select Count'))
        form.select_count_label.setVisible(True)
        form.select_count_spin.setVisible(True)

        if form.filter_widget:
            form.filter_widget.setTitle(tr('Mekraid Target'))
            form.filter_widget.set_allowed_fields(['civilizations', 'types', 'races'])

    def load_data(self, form, data):
        form.val1_spin.setValue(data.get('max_cost', 0))
        form.val2_spin.setValue(data.get('look_count', 3))
        try:
            form.select_count_spin.setValue(int(data.get('select_count', 1)))
        except Exception:
            pass

    def save_data(self, form, data):
        data['max_cost'] = form.val1_spin.value()
        data['look_count'] = form.val2_spin.value()
        try:
            data['select_count'] = int(form.select_count_spin.value())
        except Exception:
            data['select_count'] = 1


class ChoiceStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        form.option_count_label.setVisible(True)
        form.option_count_spin.setVisible(True)
        form.generate_options_btn.setVisible(True)
        form.allow_duplicates_label.setVisible(self.command_type not in ('IF', 'IF_ELSE'))
        form.allow_duplicates_check.setVisible(self.command_type not in ('IF', 'IF_ELSE'))


class LookAndAddStrategy(CommandUIStrategy):
    def load_data(self, form, data):
        form.val1_spin.setValue(data.get('look_count', 0))
        form.val2_spin.setValue(data.get('add_count', 0))

    def save_data(self, form, data):
        data['look_count'] = form.val1_spin.value()
        data['add_count'] = form.val2_spin.value()


class SelectNumberStrategy(CommandUIStrategy):
    def load_data(self, form, data):
        form.val1_spin.setValue(data.get('max', 0))
        form.val2_spin.setValue(data.get('min', 0))

    def save_data(self, form, data):
        data['max'] = form.val1_spin.value()
        data['min'] = form.val2_spin.value()


class RevolutionChangeStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        if form.filter_widget:
             form.filter_widget.setTitle(tr('Revolution Change Condition'))
             form.filter_widget.set_allowed_fields(['civilizations', 'races', 'types', 'min_cost', 'max_cost'])

    def save_data(self, form, data):
        data['type'] = 'MUTATE'
        data['mutation_kind'] = 'REVOLUTION_CHANGE'


class FriendBurstStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        if form.filter_widget:
            form.filter_widget.setTitle(tr('Friend Burst Target'))
            form.filter_widget.set_allowed_fields(['civilizations', 'races', 'types'])


class CastSpellStrategy(CommandUIStrategy):
    def update_visibility(self, form):
        super().update_visibility(form)
        if form.filter_widget:
             form.filter_widget.setTitle(tr('Spell Filter'))


STRATEGY_MAP = {
    'QUERY': QueryStrategy,
    'MUTATE': MutateStrategy,
    'MEKRAID': MekraidStrategy,
    'CHOICE': ChoiceStrategy,
    'SELECT_OPTION': ChoiceStrategy,
    'IF': ChoiceStrategy,
    'IF_ELSE': ChoiceStrategy,
    'LOOK_AND_ADD': LookAndAddStrategy,
    'SELECT_NUMBER': SelectNumberStrategy,
    'REVOLUTION_CHANGE': RevolutionChangeStrategy,
    'FRIEND_BURST': FriendBurstStrategy,
    'CAST_SPELL': CastSpellStrategy
}

def get_strategy(command_type):
    return STRATEGY_MAP.get(command_type, DefaultStrategy)(command_type)
