"""
Test that forms display only saved data, not defaults.
"""
import pytest
from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class DummyItem:
    def __init__(self, data):
        self._data = data

    def data(self, role=None):
        return self._data

    def parent(self):
        return None

    def row(self):
        return 0


class TestFormSavedDataDisplay:
    """Test that forms show saved values only, not defaults."""

    def test_empty_command_shows_no_defaults(self):
        """Test that empty command shows no default values (combos show '---')."""
        form = UnifiedActionForm()
        
        # Create a minimal ADD_KEYWORD command without optional fields
        cmd = CommandModel(
            type='ADD_KEYWORD',
            params={}
        )
        
        # Load the command via internal method (no tree item needed for unit test)
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Check that optional fields are not set (should show "---" in UI)
        # str_val and duration are optional fields with default=None
        if 'str_val' in form.widgets_map:
            widget = form.widgets_map['str_val']
            # The widget should be at initial state (empty item selected)
            if hasattr(widget, 'currentText'):
                current_text = widget.currentText()
                # Should show "---" (empty item) when no data saved
                assert current_text == "---" or current_text == "", \
                    f"str_val should show empty item, got: {current_text}"

    def test_saved_values_display_correctly(self):
        """Test that saved values are displayed correctly."""
        form = UnifiedActionForm()
        
        # Create ADD_KEYWORD with saved values
        cmd = CommandModel(
            type='ADD_KEYWORD',
            params={
                'str_val': 'S_TRIGGER',
                'duration': 'PERMANENT'
            }
        )
        
        # Load the command
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Check that saved values are displayed
        if 'str_val' in form.widgets_map:
            widget = form.widgets_map['str_val']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value == 'S_TRIGGER', f"str_val should show S_TRIGGER, got: {value}"
        
        if 'duration' in form.widgets_map:
            widget = form.widgets_map['duration']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value == 'PERMANENT', f"duration should show PERMANENT, got: {value}"

    def test_boolean_false_is_valid_value(self):
        """Test that boolean False is treated as a valid saved value."""
        form = UnifiedActionForm()
        
        # Create TRANSITION with up_to=False (explicitly set)
        cmd = CommandModel(
            type='TRANSITION',
            params={
                'zone_from': 'FIELD',
                'zone_to': 'HAND',
                'up_to': False  # False is a valid value, not "unset"
            }
        )
        
        # Load the command
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Check that up_to=False is displayed (not empty)
        if 'up_to' in form.widgets_map:
            widget = form.widgets_map['up_to']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value is False, f"up_to should be False, got: {value}"

    def test_numeric_zero_is_valid_value(self):
        """Test that numeric 0 is treated as a valid saved value."""
        form = UnifiedActionForm()
        
        # Create DRAW with count=0 (explicitly set)
        cmd = CommandModel(
            type='DRAW',
            params={
                'count': 0  # 0 is a valid value, not "unset"
            }
        )
        
        # Load the command
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Check that count=0 is displayed (not empty)
        if 'count' in form.widgets_map:
            widget = form.widgets_map['count']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value == 0, f"count should be 0, got: {value}"

    def test_partial_data_saves_correctly(self):
        """Test that partially saved data displays correctly."""
        form = UnifiedActionForm()
        
        # Create ADD_KEYWORD with only str_val set (duration unset)
        cmd = CommandModel(
            type='ADD_KEYWORD',
            params={
                'str_val': 'BLOCKER'
                # duration is intentionally omitted
            }
        )
        
        # Load the command
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Check str_val is set
        if 'str_val' in form.widgets_map:
            widget = form.widgets_map['str_val']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value == 'BLOCKER', f"str_val should show BLOCKER, got: {value}"
        
        # Check duration is not set (should show "---")
        if 'duration' in form.widgets_map:
            widget = form.widgets_map['duration']
            if hasattr(widget, 'currentText'):
                current_text = widget.currentText()
                # Should show empty item when not saved
                assert current_text == "---" or current_text == "", \
                    f"duration should show empty item, got: {current_text}"

    def test_reload_preserves_saved_state(self):
        """Test that reloading a command preserves saved state."""
        form = UnifiedActionForm()
        
        # Create, save, and reload
        cmd = CommandModel(
            type='ADD_KEYWORD',
            params={
                'str_val': 'SPEED_ATTACKER'
            }
        )
        
        # Load first time
        data = cmd.model_dump()
        item = DummyItem(data)
        form._load_ui_from_data(data, item)
        
        # Simulate persisted data by reusing the loaded data
        cmd2 = CommandModel(**data)
        
        # Load again
        data2 = cmd2.model_dump()
        item2 = DummyItem(data2)
        form._load_ui_from_data(data2, item2)
        
        # Check value is still set
        if 'str_val' in form.widgets_map:
            widget = form.widgets_map['str_val']
            if hasattr(widget, 'get_value'):
                value = widget.get_value()
                assert value == 'SPEED_ATTACKER', \
                    f"str_val should still show SPEED_ATTACKER after reload, got: {value}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
