from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget


def test_create_filter_widget_passes_only_allowed_sections(monkeypatch):
    captured = {}

    def fake_set_visible_sections(self, sections):
        # Capture the sections argument for inspection
        captured['sections'] = sections

    monkeypatch.setattr(FilterEditorWidget, 'set_visible_sections', fake_set_visible_sections)

    # Create widget for STATIC ability; sanitized sections should be captured
    widget = UnifiedFilterHandler.create_filter_widget('STATIC')
    assert 'sections' in captured
    # Only allowed keys should be present
    assert set(captured['sections'].keys()) <= {'basic', 'stats', 'flags', 'selection'}

    # Repeat for TRIGGER
    captured.clear()
    widget = UnifiedFilterHandler.create_filter_widget('TRIGGER')
    assert 'sections' in captured
    assert set(captured['sections'].keys()) <= {'basic', 'stats', 'flags', 'selection'}
