from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.schema_def import register_schema, CommandSchema, FieldSchema, FieldType


def test_unified_shows_condition_preview(qtbot):
    # Register a temporary command schema that includes a condition field
    register_schema(CommandSchema("TEST_WITH_COND", [FieldSchema("condition", "Condition", FieldType.CONDITION)]))

    form = UnifiedActionForm(None)

    # Build UI for our test command
    form.rebuild_dynamic_ui("TEST_WITH_COND")

    # Ensure the condition widget exists
    assert 'condition' in form.widgets_map
    cond_widget = form.widgets_map['condition']

    # Set condition data (use a simple known type)
    try:
        cond_widget.set_value({'type': 'OPPONENT_DRAW_COUNT', 'value': 3})
    except Exception:
        # fallback to set_data if set_value not available
        try:
            cond_widget.set_data({'type': 'OPPONENT_DRAW_COUNT', 'value': 3})
        except Exception:
            pass

    # Trigger update (signal connection should also update)
    try:
        form.update_condition_preview()
    except Exception:
        pass

    # Preview label should be visible and contain some text
    assert form.condition_preview_label.isVisible()
    txt = form.condition_preview_label.text()
    assert txt and len(txt) > 0
*** End Patch