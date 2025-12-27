from ..QtCore import Qt
from ..QtGui import QAction

class _Signal:
    def __init__(self):
        self._callbacks = []
    def connect(self, cb):
        try:
            self._callbacks.append(cb)
        except Exception:
            pass

class QApplication:
    def __init__(self, argv=None):
        pass
    def exec(self):
        return 0

class QMainWindow:
    def __init__(self, *args, **kwargs):
        pass
    def setWindowTitle(self, t):
        pass
    def resize(self, w, h):
        pass
    def addToolBar(self, tb):
        pass
    def addDockWidget(self, area, dock):
        pass
    def setCentralWidget(self, w):
        self._central = w
    def splitDockWidget(self, first, second, orientation):
        # Layout management is not simulated; record arrangement
        self._split = (first, second, orientation)
    def show(self):
        self._visible = True
    def showMaximized(self):
        self._maximized = True

class QWidget:
    def __init__(self, *args, **kwargs):
        pass
    def setMinimumWidth(self, w):
        pass
    def setLayout(self, layout):
        self._layout = layout
    def setFixedSize(self, w, h):
        self._fixed_size = (w, h)
    def setParent(self, p):
        self._parent = p
    def setCursor(self, cursor):
        self._cursor = cursor
    def setAccessibleName(self, name):
        self._accessible_name = name
    def setAccessibleDescription(self, desc):
        self._accessible_description = desc
    def setToolTip(self, tip):
        self._tooltip = tip
    def setStyleSheet(self, s: str):
        self._style = s

class QVBoxLayout:
    def __init__(self, parent=None):
        self._items = []
    def addLayout(self, l):
        self._items.append(l)
    def addWidget(self, w, *args, **kwargs):
        # Accept optional stretch/alignment kwargs used in real Qt
        self._items.append(w)
    def setContentsMargins(self, l, t, r, b):
        pass
    def setSpacing(self, s):
        pass
    def addStretch(self, v=0):
        pass
    def setAlignment(self, alignment):
        self._alignment = alignment
    def count(self):
        return len(self._items)
    def itemAt(self, index):
        try:
            w = self._items[index]
        except Exception:
            return None
        class _Item:
            def __init__(self, w):
                self._w = w
            def widget(self):
                return self._w
        return _Item(w)

class QHBoxLayout(QVBoxLayout):
    pass

class QLabel:
    def __init__(self, text=''):
        self._text = text
    def setStyleSheet(self, s):
        pass
    def setWordWrap(self, flag: bool):
        pass
    def setFixedWidth(self, w):
        pass
    def setFixedHeight(self, h):
        pass
    def setAlignment(self, alignment):
        pass
    def setText(self, text):
        try:
            self._text = str(text)
        except Exception:
            self._text = ''
    def text(self):
        return getattr(self, '_text', '')
    def setFixedSize(self, w, h):
        self._fixed_size = (w, h)
    def font(self):
        try:
            from ..QtGui import QFont
            return QFont()
        except Exception:
            return None
    def setFont(self, font):
        self._font = font
    def setVisible(self, flag: bool):
        self._visible = bool(flag)

class QProgressBar:
    def __init__(self):
        self._min = 0
        self._max = 100
        self._value = 0
        self._format = ''
    def setRange(self, lo, hi):
        self._min = lo
        self._max = hi
    def setFormat(self, fmt):
        self._format = fmt
    def setValue(self, v):
        try:
            self._value = int(v)
        except Exception:
            pass

class QPushButton:
    def __init__(self, text=''):
        self._text = text
        self.clicked = _Signal()
    def setToolTip(self, txt):
        pass
    def setShortcut(self, sc):
        pass
    def clicked_connect(self, cb):
        self.clicked.connect(cb)
    def setVisible(self, flag: bool):
        pass
    def setStyleSheet(self, s: str):
        pass
    def setEnabled(self, flag: bool):
        self._enabled = bool(flag)
    def setCheckable(self, flag: bool):
        self._checkable = bool(flag)

class QListWidget:
    def __init__(self):
        self._items = []
        self._drag_drop_mode = None
        class _Model:
            def __init__(self):
                self.rowsMoved = Signal()
        self._model = _Model()
    def setDragDropMode(self, mode):
        self._drag_drop_mode = mode
    def addItem(self, item):
        self._items.append(item)
    def insertItem(self, index, item):
        try:
            self._items.insert(index, item)
        except Exception:
            self._items.append(item)
    def clear(self):
        self._items = []
    def model(self):
        return self._model
    def selectedItems(self):
        # No selection handling in stub; return empty list
        return []

class QAbstractItemView:
    class DragDropMode:
        InternalMove = 0
        NoDragDrop = 1

class QTextEdit:
    def __init__(self, *args, **kwargs):
        self._text = ''
        self._readonly = False
    def setReadOnly(self, flag: bool):
        self._readonly = bool(flag)
    def setPlainText(self, txt: str):
        self._text = str(txt)
    def toPlainText(self):
        return str(self._text)
    def setPlaceholderText(self, txt: str):
        try:
            self._placeholder = str(txt)
        except Exception:
            self._placeholder = ''

class QFileDialog:
    @staticmethod
    def getOpenFileName(*args, **kwargs):
        return ('', '')

class QMessageBox:
    @staticmethod
    def information(parent, title, text):
        return None

class QSplitter:
    def __init__(self, orientation=None, parent=None):
        self._orientation = orientation
        self._parent = parent
        self._widgets = []
    def addWidget(self, w):
        self._widgets.append(w)

class QHeaderView:
    class ResizeMode:
        ResizeToContents = 0
        Stretch = 1
    def __init__(self):
        pass
    def setSectionResizeMode(self, section, mode):
        self._resize_modes = getattr(self, '_resize_modes', {})
        self._resize_modes[section] = mode

class QTreeWidgetItem:
    def __init__(self, columns=None):
        self._columns = list(columns) if columns is not None else []
        self._children = []
    def addChild(self, item):
        self._children.append(item)
    def setText(self, idx, text):
        try:
            self._columns[idx] = text
        except Exception:
            pass
    def setData(self, role, value):
        self._data = getattr(self, '_data', {})
        self._data[role] = value

class QTreeWidget:
    def __init__(self):
        self._header = QHeaderView()
        self._items = []
    def setHeaderLabels(self, labels):
        self._headers = list(labels)
    def header(self):
        return self._header
    def setColumnWidth(self, col, w):
        pass
    def addTopLevelItem(self, item):
        self._items.append(item)
    def clear(self):
        self._items = []

class QGraphicsEllipseItem:
    def __init__(self, *args, **kwargs):
        self._tooltip = ''
        self._z = 0
    def setToolTip(self, t):
        self._tooltip = t
    def setZValue(self, v):
        self._z = v

class QGraphicsLineItem:
    def __init__(self, *args, **kwargs):
        self._pen = None
    def setPen(self, pen):
        self._pen = pen
    def setZValue(self, v):
        self._z = v

class QGraphicsTextItem:
    def __init__(self, text=''):
        self._text = text
    def setFont(self, font):
        self._font = font
    def boundingRect(self):
        class _R:
            def width(self):
                return 50
            def height(self):
                return 10
        return _R()
    def setPos(self, x, y):
        self._pos = (x, y)

class QGraphicsScene:
    def __init__(self, parent=None):
        self._items = []
        self._parent = parent
    def addEllipse(self, rect, pen=None, brush=None):
        item = QGraphicsEllipseItem()
        self._items.append(item)
        return item
    def addItem(self, item):
        self._items.append(item)
    def clear(self):
        self._items = []

class QGraphicsView(QWidget):
    class DragMode:
        NoDrag = 0
        ScrollHandDrag = 1
    class ViewportAnchor:
        class AnchorUnderMouse:
            AnchorUnderMouse = 0
    def __init__(self, parent=None):
        super().__init__()
        self._scene = None
    def setScene(self, scene):
        self._scene = scene
    def setRenderHint(self, hint):
        pass
    def setDragMode(self, mode):
        self._drag_mode = mode
    def setTransformationAnchor(self, anchor):
        self._anchor = anchor

class QCheckBox:
    def __init__(self, text=''):
        self._text = text
        self._checked = False
        self.stateChanged = Signal()
    def setChecked(self, flag: bool):
        self._checked = bool(flag)
    def isChecked(self):
        return bool(getattr(self, '_checked', False))

class Signal:
    def __init__(self):
        self._slots = []
    def connect(self, fn):
        self._slots.append(fn)
    def disconnect(self, fn):
        try:
            self._slots.remove(fn)
        except ValueError:
            pass
    def emit(self, *args, **kwargs):
        for s in list(self._slots):
            try:
                s(*args, **kwargs)
            except Exception:
                pass

class QGroupBox(QWidget):
    def __init__(self, title=''):
        pass
    def setLayout(self, layout):
        self._layout = layout

class QFrame(QWidget):
    class Shape:
        Box = 1
    class Shadow:
        Raised = 1
    def __init__(self, *args, **kwargs):
        super().__init__()
    def setFrameStyle(self, style):
        self._frame_style = style
    def setLineWidth(self, w):
        self._line_width = w

class QRadioButton:
    def __init__(self, text=''):
        self._text = text
        self._checked = False
        self.stateChanged = Signal()
    def setChecked(self, flag: bool):
        self._checked = bool(flag)
    def isChecked(self):
        return bool(getattr(self, '_checked', False))

class QButtonGroup:
    def __init__(self):
        pass
    def addButton(self, btn):
        pass

class QScrollArea(QWidget):
    def __init__(self):
        self._widget = None
        self._resizable = False
        self._min_height = None
    def setWidgetResizable(self, flag: bool):
        self._resizable = bool(flag)
    def setMinimumHeight(self, h: int):
        try:
            self._min_height = int(h)
        except Exception:
            self._min_height = None
    def setVerticalScrollBarPolicy(self, policy):
        self._vscroll_policy = policy
    def setWidget(self, w):
        self._widget = w

class QDockWidget(QWidget):
    def __init__(self, title='', parent=None):
        pass
    def setObjectName(self, name):
        pass
    def setAllowedAreas(self, areas):
        pass
    def setWidget(self, w):
        pass
    def hide(self):
        self._hidden = True
    def show(self):
        self._hidden = False

class QTabWidget(QWidget):
    def __init__(self):
        self._tabs = []
    def addTab(self, widget, label=''):
        self._tabs.append((widget, label))
        return len(self._tabs)-1

class QInputDialog:
    pass

class QComboBox:
    def __init__(self):
        self._items = []  # list of (text, data)
        self._current = 0
    def addItem(self, text, data=None):
        self._items.append((text, data))
    def clear(self):
        self._items = []
        self._current = 0
    def currentData(self):
        try:
            return self._items[self._current][1]
        except Exception:
            return None
    def setCurrentIndex(self, idx):
        try:
            self._current = int(idx)
        except Exception:
            pass

class QSpinBox:
    def __init__(self):
        self._min = 0
        self._max = 100
        self._value = 0
    def setRange(self, lo, hi):
        self._min = lo
        self._max = hi
    def setValue(self, v):
        try:
            self._value = int(v)
        except Exception:
            pass

class QFormLayout:
    def __init__(self):
        self._rows = []
    def addWidget(self, w):
        # Many calls add widgets directly; accept single widget
        self._rows.append((w,))
    def addRow(self, label, widget):
        self._rows.append((label, widget))

class QToolBar:
    def __init__(self, title='', parent=None):
        self._title = title
        self._parent = parent
    def setObjectName(self, name):
        self._name = name
    def addAction(self, act):
        pass
    def setVisible(self, flag):
        pass

__all__ = [
    'QApplication','QMainWindow','QWidget','QVBoxLayout','QHBoxLayout','QLabel','QPushButton',
    'QListWidget','QFileDialog','QMessageBox','QSplitter','QCheckBox','QGroupBox','QRadioButton',
    'QButtonGroup','QScrollArea','QDockWidget','QTabWidget','QInputDialog','QToolBar'
]


def _make_dummy(name):
    class _D:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, item):
            return None
    _D.__name__ = name
    return _D

# Provide common widget names used across the GUI as fallbacks
_common = [
    'QLineEdit','QTextEdit','QTreeWidget','QTreeWidgetItem','QProgressBar','QHeaderView',
    'QTreeWidgetItem','QSize','QGraphicsView','QGraphicsScene','QGraphicsEllipseItem',
    'QGraphicsLineItem','QGraphicsTextItem','QMimeData','QDrag','QTabBar','QStackedWidget',
    'QAbstractItemView','QListWidgetItem'
]
for _n in _common:
    globals().setdefault(_n, _make_dummy(_n))

def __getattr__(name: str):
    # Return a dummy class for any missing attribute to maximize import compatibility
    if name.startswith('_'):
        raise AttributeError(name)
    val = globals().get(name)
    if val is not None:
        return val
    dummy = _make_dummy(name)
    globals()[name] = dummy
    return dummy
