# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget

class ConditionTreeWidget(ConditionEditorWidget):
    """
    Temporary implementation of ConditionTreeWidget to satisfy Schema requirements.
    Currently behaves identically to ConditionEditorWidget (single node).
    Future improvements should implement actual tree/nesting logic.
    """
    def __init__(self, parent=None):
        super().__init__(parent, title="Condition")
