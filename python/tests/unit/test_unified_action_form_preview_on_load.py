from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.schema_def import register_schema, CommandSchema, FieldSchema, FieldType


def test_preview_shows_on_load():
    # Register a temporary command schema that includes a condition field
    register_schema(CommandSchema("TEST_WITH_COND_LOAD", [FieldSchema("condition", "Condition", FieldType.CONDITION)]))

    form = UnifiedActionForm(None)

    # Prepare command data with condition at top-level (ingest will move to params)
    data = {
        'type': 'TEST_WITH_COND_LOAD',
        'condition': {'type': 'OPPONENT_DRAW_COUNT', 'value': 4}
    }

    # Load UI from data
    form._load_ui_from_data(data, None)

    # Preview label should be visible and contain some non-empty text
    try:
        visible = form.condition_preview_label.isVisible()
    except Exception:
        visible = True if form.condition_preview_label.text() else False

    assert visible
    # Also assert the underlying widget produces a preview
    cond_widget = form.widgets_map.get('condition')
    assert cond_widget is not None
    widget_preview = ''
    try:
        widget_preview = cond_widget.get_preview_text()
    except Exception:
        widget_preview = ''

    assert widget_preview and len(widget_preview) > 0, f"Widget preview empty: {widget_preview}"

    # Ensure unified preview label reflects the widget preview (try updating once)
    try:
        form.update_condition_preview()
    except Exception:
        pass
    txt = form.condition_preview_label.text()
    assert txt and len(txt) > 0
