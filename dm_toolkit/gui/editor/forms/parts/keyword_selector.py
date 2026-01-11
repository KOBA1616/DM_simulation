# -*- coding: utf-8 -*-
"""
Unified Keyword Selector Widget for Static Abilities.
Used by both GRANT_KEYWORD and SET_KEYWORD modifiers.
"""

from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.editor.text_resources import CardTextResources


class KeywordSelectorWidget(QComboBox):
    """
    Unified keyword selector for static ability modifiers.
    Provides Japanese display names with English enum values as data.
    """
    
    # Signal emitted when keyword is selected
    keywordSelected = pyqtSignal(str)
    
    def __init__(self, allow_settable: bool = True, parent=None):
        """
        Initialize keyword selector.
        
        Args:
            allow_settable: If True, include both GRANT and SET keywords.
                          If False, only include GRANT keywords.
            parent: Parent widget
        """
        super().__init__(parent)
        self.allow_settable = allow_settable
        self._populate_keywords()
        
        # Connect selection change signal
        self.currentIndexChanged.connect(self._on_selection_changed)
    
    def _populate_keywords(self):
        """Populate combobox with keywords in Japanese."""
        from dm_toolkit.consts import GRANTABLE_KEYWORDS, SETTABLE_KEYWORDS
        
        # Clear existing items
        self.blockSignals(True)
        self.clear()
        
        # Add grantable keywords
        for keyword in GRANTABLE_KEYWORDS:
            japanese_text = CardTextResources.get_keyword_text(keyword)
            self.addItem(japanese_text, keyword)
        
        # Add settable keywords if enabled
        if self.allow_settable and SETTABLE_KEYWORDS:
            for keyword in SETTABLE_KEYWORDS:
                japanese_text = CardTextResources.get_keyword_text(keyword)
                self.addItem(japanese_text, keyword)
        
        self.blockSignals(False)
    
    def set_keyword(self, keyword: str):
        """
        Set the current keyword by enum value.
        
        Args:
            keyword: Keyword enum string (e.g., "BLOCKER")
        """
        for i in range(self.count()):
            if self.itemData(i) == keyword:
                self.blockSignals(True)
                self.setCurrentIndex(i)
                self.blockSignals(False)
                return
    
    def get_keyword(self) -> str:
        """
        Get the currently selected keyword enum value.
        
        Returns:
            Selected keyword string, or empty string if none selected
        """
        return self.currentData() or ""
    
    def _on_selection_changed(self, index: int):
        """Signal emitted when selection changes."""
        keyword = self.itemData(index)
        if keyword:
            self.keywordSelected.emit(keyword)
