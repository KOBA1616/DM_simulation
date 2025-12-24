import sys
import os

# Mock modules
sys.modules['dm_ai_module'] = type('dm_ai_module', (), {})

class MockFont:
    def setBold(self, *args): pass
    def setPointSize(self, *args): pass

class MockQLabel:
    def __init__(self, *args, **kwargs): pass
    def setAlignment(self, *args): pass
    def setStyleSheet(self, *args): pass
    def setFixedSize(self, *args): pass
    def update(self): pass
    def setVisible(self, *args): pass
    def font(self): return MockFont()
    def setFont(self, *args): pass
    def setWordWrap(self, *args): pass
    def setText(self, *args): pass
    def setFixedHeight(self, *args): pass
    def setReadOnly(self, *args): pass
    def clear(self): pass
    def rect(self): return type('Rect', (), {'width': lambda: 10, 'height': lambda: 10, 'adjusted': lambda *a: type('Rect', (), {'width': lambda: 10})() })()

class MockQWidget:
    def __init__(self, *args, **kwargs): pass
    def hide(self): pass
    def show(self): pass
    def setFixedHeight(self, *args): pass
    def setReadOnly(self, *args): pass

class MockFrame:
    class Shape:
        StyledPanel=1
    class Shadow:
        Raised=1
    def __init__(self, *args, **kwargs): pass
    def setFrameShape(self, *args): pass
    def setFrameShadow(self, *args): pass
    def setLineWidth(self, *args): pass
    def setFixedSize(self, *args): pass
    def setStyleSheet(self, *args): pass

class MockLayout:
    def __init__(self, *args): pass
    def setContentsMargins(self, *args): pass
    def addWidget(self, *args): pass
    def addLayout(self, *args): pass
    def setSpacing(self, *args): pass
    def addStretch(self): pass
    def setRowStretch(self, *args): pass
    def setStretch(self, *args): pass
    def addSpacing(self, *args): pass

# Mock PyQt6
mock_qt = type('Qt', (), {'AlignmentFlag': type('Flags', (), {'AlignCenter': 0, 'AlignLeft': 0, 'AlignRight': 0, 'AlignVCenter': 0, 'AlignTop': 0, 'AlignBottom': 0}), 'GlobalColor': type('GC', (), {'black': 0}), 'PenStyle': type('PS', (), {'NoPen': 0}), 'BrushStyle': type('BS', (), {'NoBrush': 0})})
sys.modules['PyQt6.QtCore'] = type('QtCore', (), {'Qt': mock_qt})
sys.modules['PyQt6.QtGui'] = type('QtGui', (), {'QFont': MockFont, 'QColor': lambda x: None, 'QPainter': type('Painter', (), {'RenderHint': type('RH', (), {'Antialiasing': 0})}), 'QPen': lambda x: type('Pen', (), {'setWidth': lambda s:None})()})
sys.modules['PyQt6.QtWidgets'] = type('QtWidgets', (), {
    'QWidget': MockQWidget, 'QVBoxLayout': MockLayout, 'QLabel': MockQLabel,
    'QGroupBox': MockQWidget, 'QTextEdit': MockQWidget, 'QFrame': MockFrame,
    'QGridLayout': MockLayout, 'QHBoxLayout': MockLayout,
    'QGraphicsDropShadowEffect': MockQWidget
})

sys.path.append(os.getcwd())
from dm_toolkit.gui.editor.preview_pane import CardPreviewWidget, ManaCostLabel

def test_apply_cost_circle_style():
    w = CardPreviewWidget()
    # Mock labels
    lbl = ManaCostLabel("5")
    # This should work (no error)
    w.apply_cost_circle_style(lbl, ["FIRE"])
    print("ManaCostLabel style applied successfully")

    # This should do NOTHING now (no fallback)
    lbl2 = MockQLabel("Test")
    w.apply_cost_circle_style(lbl2, ["FIRE"])
    print("Standard QLabel skipped successfully")

if __name__ == "__main__":
    test_apply_cost_circle_style()
