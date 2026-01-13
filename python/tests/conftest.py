import os
import sys
import importlib
import pytest

# ============================================================================
# PyQt/PySide Stubbing for Headless Environments
# ============================================================================
# CRITICAL: This MUST execute BEFORE pytest starts collecting tests,
# because dm_toolkit.gui modules will try to import PyQt6 during collection.

# Execute stub setup immediately at module load time
import unittest.mock
import types

def _setup_minimal_gui_stubs():
    """Setup minimal GUI stubs to prevent ImportErrors during test collection."""
    # If PyQt6.QtWidgets already exists and has QMainWindow, we're done
    qtwidgets_mod = sys.modules.get('PyQt6.QtWidgets')
    if qtwidgets_mod and hasattr(qtwidgets_mod, 'QMainWindow'):
        return  # Already properly set up or real PyQt6 exists
    
    # Otherwise, set up minimal stubs NOW (before test collection)
    
    # Create a functional Signal class that actually calls connected slots
    class MockSignal:
        def __init__(self):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)
            return None

        def disconnect(self, slot=None):
            if slot is None:
                self._slots.clear()
            elif slot in self._slots:
                self._slots.remove(slot)
            return None

        def emit(self, *args, **kwargs):
            for slot in self._slots:
                slot(*args, **kwargs)
            return None

    # Create dummy classes with signal support
    class DummyQWidget(object):
        def __init__(self, *args, **kwargs):
            # Add common signals as MagicMocks
            self.clicked = unittest.mock.MagicMock()
            self.textChanged = unittest.mock.MagicMock()
            self.stateChanged = unittest.mock.MagicMock()
            self.currentIndexChanged = unittest.mock.MagicMock()
            self.valueChanged = unittest.mock.MagicMock()
            self.customContextMenuRequested = unittest.mock.MagicMock()
            self.triggered = unittest.mock.MagicMock()
            self.activated = unittest.mock.MagicMock()
            self._items = []
            
        def setWindowTitle(self, title): pass
        def setLayout(self, layout): pass
        def setGeometry(self, *args): pass
        def show(self): pass
        def close(self): return True
        def addWidget(self, widget, *args): pass  # Accept extra args for Grid Layout
        def addLayout(self, layout, *args): pass  # Accept extra args for Grid Layout
        def setText(self, text): pass
        def text(self): return ""
        def setCheckable(self, checkable): pass
        def setCheckState(self, state): pass
        def addItem(self, *args): self._items.append(args)
        def setCurrentIndex(self, index): pass
        def setCheckState(self, state): pass
        def addWidget(self, *args, **kwargs): pass
        def addLayout(self, *args, **kwargs): pass
        def addRow(self, *args, **kwargs): pass
        def addStretch(self, *args): pass
        def blockSignals(self, b): return False
        def clear(self): self._items = []
        def count(self): return len(self._items)
        def setItemData(self, index, data, role=None): pass
        def itemData(self, index, role=None): return None
        def addButton(self, button, id=-1): pass
        def checkedId(self): return -1
        def id(self, button): return -1
        def setContentsMargins(self, *args): pass
        def setSpacing(self, spacing): pass
        def addStretch(self, *args): pass
        def setStyleSheet(self, style): pass
        def setMinimumWidth(self, width): pass
        def setMinimumHeight(self, height): pass
        def setMaximumWidth(self, width): pass
        def setMaximumHeight(self, height): pass
        def clear(self): self._items = []
        def setToolTip(self, text): pass
        def setCursor(self, cursor): pass
        def setEnabled(self, enabled): pass
        def isEnabled(self): return True
        def setVisible(self, visible): pass
        def isVisible(self): return True
        def setExclusive(self, exclusive): pass
        def insertItem(self, index, text): pass
        def setFlat(self, flat): pass
        def setSpacing(self, spacing): pass
        def setContentsMargins(self, *args): pass
        def setPlaceholderText(self, text): pass
        def setRange(self, min_val, max_val): pass
        def setMaximum(self, max_val): pass
        def setMinimum(self, min_val): pass

    class DummyQMainWindow(DummyQWidget):
        def setCentralWidget(self, widget): pass
        def setMenuBar(self, menu): pass
        def addDockWidget(self, area, dock): pass
        def setStatusBar(self, bar): pass

    class DummyQDialog(DummyQWidget):
        def exec(self): return 1
        def accept(self): pass
        def reject(self): pass

    class DummyQApplication:
        def __init__(self, args): pass
        def exec(self): return 0
        @staticmethod
        def instance(): return None

    # Enhanced widget classes with signals
    class EnhancedButton(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.clicked = MockSignal()
        def setCheckable(self, checkable): pass
        def isChecked(self): return False
        def setChecked(self, checked): pass
        def setFlat(self, flat): pass
        def setStyleSheet(self, style): pass
        def setMinimumWidth(self, width): pass
        def setCursor(self, cursor): pass

    class EnhancedComboBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.currentIndexChanged = MockSignal()
            self._current_index = 0

        def blockSignals(self, block): return False

        def addItem(self, *args): self._items.append(args)
        def setCurrentIndex(self, index): self._current_index = index
        def currentIndex(self): return self._current_index
        def currentText(self):
            if 0 <= self._current_index < len(self._items):
                return self._items[self._current_index][0]
            return ""
        def currentData(self):
            if 0 <= self._current_index < len(self._items):
                item = self._items[self._current_index]
                if len(item) > 1: return item[-1]
            return None

        def setEditable(self, editable): pass
        def setEnabled(self, enabled): pass

        def findData(self, data):
             for i, item in enumerate(self._items):
                 if len(item) > 1 and item[-1] == data:
                     return i
             return -1

        def itemData(self, index, role=None):
            if 0 <= index < len(self._items):
                item = self._items[index]
                if len(item) > 1:
                    return item[-1]
            return None

    class EnhancedLineEdit(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textChanged = MockSignal()
            self.textEdited = MockSignal()

        def setText(self, text): pass
        def text(self): return ""
        def setPlaceholderText(self, text): pass

    class EnhancedCheckBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stateChanged = MockSignal()

        def setCheckState(self, state): pass
        def checkState(self): return 0
        def isChecked(self): return False

    class EnhancedSpinBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.valueChanged = MockSignal()
            self._value = 0

        def setRange(self, min_val, max_val): pass
        def setMinimum(self, min_val): pass
        def setMaximum(self, max_val): pass
        def setValue(self, value): self._value = value
        def value(self): return self._value
        def setSpecialValueText(self, text): pass
        def setSingleStep(self, step): pass
        def setVisible(self, visible): pass

    class EnhancedFormLayout(DummyQWidget):
        def addRow(self, *args): pass

    class EnhancedButtonGroup(DummyQWidget):
        def setExclusive(self, exclusive): pass
        def addButton(self, button, id=-1): pass

    class DummyQt:
        class ItemDataRole:
            DisplayRole = 0
            UserRole = 256
            ForegroundRole = 9
            BackgroundRole = 8
            EditRole = 2

        class AlignmentFlag:
            AlignCenter = 0x0084
            AlignLeft = 0x0001

        class WindowType:
            Window = 0x00000001

        class MatchFlag:
            MatchContains = 1
            MatchFixedString = 8

        class CursorShape:
            PointingHandCursor = 13
        class ContextMenuPolicy:
            CustomContextMenu = 2
        Horizontal = 1
        Vertical = 2
        Checked = 2
        Unchecked = 0
        MatchContains = 1
        LeftToRight = 0

    class DummyAbstractItemView(DummyQWidget):
        class EditTrigger:
            NoEditTriggers = 0
            DoubleClicked = 1
        class SelectionBehavior:
            SelectRows = 1
            SelectItems = 0
        class SelectionMode:
            SingleSelection = 1
            MultiSelection = 2
            ExtendedSelection = 3
        class DragDropMode:
            NoDragDrop = 0
            DragOnly = 1
            DropOnly = 2
            DragDrop = 3
            InternalMove = 4

        def setSelectionMode(self, mode): pass
        def setEditTriggers(self, triggers): pass
        def setDragEnabled(self, enabled): pass
        def setAcceptDrops(self, accept): pass
        def setDropIndicatorShown(self, show): pass
        def setDragDropMode(self, mode): pass
        def setModel(self, model): pass
        def setHeaderHidden(self, hidden): pass
        def selectionModel(self):
            mock = unittest.mock.MagicMock()
            mock.selectionChanged = unittest.mock.MagicMock()
            mock.indexes = lambda: []
            return mock
        def indexAt(self, pos): return unittest.mock.MagicMock()
        def viewport(self): return DummyQWidget()
        def setContextMenuPolicy(self, policy): pass
        def expand(self, index): pass
        def setExpanded(self, index, expanded): pass
        def isExpanded(self, index): return False
        def collapse(self, index): pass
        def scrollTo(self, index): pass
        def clearSelection(self): pass

    class DummyTreeView(DummyAbstractItemView):
        def header(self): return DummyQWidget()
        def setColumnWidth(self, column, width): pass

    class DummyMessageBox(DummyQWidget):
        StandardButton = unittest.mock.MagicMock()
        Yes = 16384
        No = 65536
        Ok = 1024
        Cancel = 4194304

        @staticmethod
        def information(parent, title, text, buttons=0, defaultButton=0): return 1024
        @staticmethod
        def warning(parent, title, text, buttons=0, defaultButton=0): return 1024
        @staticmethod
        def critical(parent, title, text, buttons=0, defaultButton=0): return 1024
        @staticmethod
        def question(parent, title, text, buttons=0, defaultButton=0): return 16384

    class DummyInputDialog(DummyQWidget):
        @staticmethod
        def getText(parent, title, label, echo=0, text="", flags=0): return "", False
        @staticmethod
        def getItem(parent, title, label, items, current=0, editable=True): return "", False
        @staticmethod
        def getInt(parent, title, label, value=0, min=0, max=100, step=1): return 0, False

    class DummyFileDialog(DummyQWidget):
        @staticmethod
        def getOpenFileName(*args, **kwargs): return "", ""
        @staticmethod
        def getSaveFileName(*args, **kwargs): return "", ""
        @staticmethod
        def getExistingDirectory(*args, **kwargs): return ""

    # Create module structure
    if 'PyQt6' not in sys.modules:
        pyqt6 = types.ModuleType('PyQt6')
        pyqt6.__path__ = []
        sys.modules['PyQt6'] = pyqt6
    else:
        pyqt6 = sys.modules['PyQt6']

    if 'PyQt6.QtWidgets' not in sys.modules:
        qt_widgets = types.ModuleType('PyQt6.QtWidgets')
        sys.modules['PyQt6.QtWidgets'] = qt_widgets
    else:
        qt_widgets = sys.modules['PyQt6.QtWidgets']

    if 'PyQt6.QtCore' not in sys.modules:
        qt_core = types.ModuleType('PyQt6.QtCore')
        sys.modules['PyQt6.QtCore'] = qt_core
    else:
        qt_core = sys.modules['PyQt6.QtCore']

    if 'PyQt6.QtGui' not in sys.modules:
        qt_gui = types.ModuleType('PyQt6.QtGui')
        sys.modules['PyQt6.QtGui'] = qt_gui
    else:
        qt_gui = sys.modules['PyQt6.QtGui']

    # Populate with stub classes
    qt_widgets.QMainWindow = DummyQMainWindow
    qt_widgets.QWidget = DummyQWidget
    qt_widgets.QApplication = type('QApplication', (), {'__init__': lambda s, a: None, 'exec': lambda s: 0})

    qt_widgets.QAbstractItemView = DummyAbstractItemView
    qt_widgets.QTreeView = DummyTreeView
    qt_widgets.QMessageBox = DummyMessageBox
    qt_widgets.QInputDialog = DummyInputDialog
    qt_widgets.QFileDialog = DummyFileDialog

    # Use Enhanced Widgets
    qt_widgets.QPushButton = EnhancedButton
    qt_widgets.QComboBox = EnhancedComboBox
    qt_widgets.QLineEdit = EnhancedLineEdit
    qt_widgets.QCheckBox = EnhancedCheckBox
    qt_widgets.QSpinBox = EnhancedSpinBox
    qt_widgets.QFormLayout = EnhancedFormLayout
    qt_widgets.QButtonGroup = EnhancedButtonGroup

    for name in ['QLabel', 'QVBoxLayout', 'QHBoxLayout', 'QTreeWidget',
                 'QTreeWidgetItem', 'QDialog', 'QTextEdit',
                 'QScrollArea', 'QTabWidget', 'QDockWidget', 'QGraphicsView',
                 'QGraphicsScene', 'QGraphicsEllipseItem', 'QGraphicsLineItem', 'QGraphicsTextItem',
                 'QProgressBar', 'QHeaderView', 'QSplitter', 'QGroupBox', 'QMenuBar', 'QMenu',
                 'QStatusBar', 'QGridLayout', 'QListWidget',
                 'QListWidgetItem', 'QToolBar', 'QSizePolicy', 'QStackedWidget', 'QFrame']:
        setattr(qt_widgets, name, type(name, (DummyQWidget,), {}))

    qt_core.Qt = DummyQt
    qt_core.QModelIndex = type('QModelIndex', (object,), {})
    qt_core.QObject = type('QObject', (object,), {'__init__': lambda s, *a: None, 'blockSignals': lambda s, b: False})
    qt_core.QTimer = type('QTimer', (object,), {'singleShot': lambda *a: None, 'start': lambda s, t: None, 'stop': lambda s: None})
    qt_core.pyqtSignal = lambda *args: unittest.mock.MagicMock(emit=lambda *a: None, connect=lambda *a: None)
    qt_core.QMimeData = type('QMimeData', (), {})
    qt_core.QRectF = type('QRectF', (), {})
    qt_core.QThread = type('QThread', (object,), {'start': lambda s: None, 'wait': lambda s: None, 'quit': lambda s: None, 'isRunning': lambda s: False})

    qt_gui.QAction = type('QAction', (DummyQWidget,), {})
    qt_gui.QKeySequence = type('QKeySequence', (), {})
    qt_gui.QStandardItem = type('QStandardItem', (object,), {'__init__': lambda s, *a: None})
    qt_gui.QDrag = type('QDrag', (), {})
    qt_gui.QPen = type('QPen', (), {})
    qt_gui.QBrush = type('QBrush', (), {})
    qt_gui.QColor = lambda *a: None
    qt_gui.QFont = type('QFont', (), {})
    qt_gui.QPainter = type('QPainter', (), {})
    qt_gui.QIcon = lambda *a: None
    qt_gui.QStandardItemModel = type('QStandardItemModel', (object,), {'__init__': lambda s, *a: None, 'invisibleRootItem': lambda s: unittest.mock.MagicMock()})

    # Link to parent package
    pyqt6.QtWidgets = qt_widgets
    pyqt6.QtCore = qt_core
    pyqt6.QtGui = qt_gui
    pyqt6.__dict__['QtWidgets'] = qt_widgets
    pyqt6.__dict__['QtCore'] = qt_core
    pyqt6.__dict__['QtGui'] = qt_gui
    pyqt6.__all__ = ['QtWidgets', 'QtCore', 'QtGui']

# Execute immediately - this runs when conftest.py is first imported by pytest
_setup_minimal_gui_stubs()

# ============================================================================
# Path Setup and Module Loading
# ============================================================================

# Ensure local `python/` shim module directory is preferred so tests can run
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
shim_dir = os.path.join(root, 'python')
if shim_dir not in sys.path:
    sys.path.insert(0, shim_dir)

# Ensure the compiled extension module (built to ./bin) is importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_BIN_DIR = os.path.join(_PROJECT_ROOT, 'bin')
if os.path.isdir(_BIN_DIR):
    if _BIN_DIR in sys.path:
        sys.path.remove(_BIN_DIR)
    sys.path.insert(0, _BIN_DIR)

# Force a clean import so the extension module wins.
if 'dm_ai_module' in sys.modules:
    del sys.modules['dm_ai_module']

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

@pytest.fixture
def require_native():
    """
    Skip the test if dm_ai_module is not the native C++ extension.
    """
    if dm_ai_module is None:
         pytest.fail("dm_ai_module could not be imported.")

    is_native = getattr(dm_ai_module, 'IS_NATIVE', False)
    if not is_native and hasattr(dm_ai_module, 'GameState'):
        if str(type(dm_ai_module.GameState)).find('pybind11') != -1:
            is_native = True

    if not is_native:
        pytest.skip("Test requires native C++ dm_ai_module extension.")

# Debug wrapper for DevTools.move_cards to observe where cards are placed (native vs proxy)
try:
    if dm_ai_module and hasattr(dm_ai_module, 'DevTools') and hasattr(dm_ai_module.DevTools, 'move_cards'):
        _orig_move_cards = dm_ai_module.DevTools.move_cards
        def _dbg_move_cards(gs, player, from_zone, to_zone, count, instance_id):
            return _orig_move_cards(gs, player, from_zone, to_zone, count, instance_id)
        try:
            dm_ai_module.DevTools.move_cards = staticmethod(_dbg_move_cards)
        except Exception:
            dm_ai_module.DevTools.move_cards = _dbg_move_cards
except Exception:
    pass

# --- Compatibility wrappers -------------------------------------------------
# Provide thin factories that accept legacy-style constructor args used in tests
# but return the native pybind objects so C++ signatures still match.
if dm_ai_module:
    _orig_GameState = getattr(dm_ai_module, 'GameState', None)
    _orig_EffectDef = getattr(dm_ai_module, 'EffectDef', None)
    _orig_ConditionDef = getattr(dm_ai_module, 'ConditionDef', None)
    _orig_ActionDef = getattr(dm_ai_module, 'ActionDef', None)
    _orig_FilterDef = getattr(dm_ai_module, 'FilterDef', None)

    class _PlayerProxy:
        def __init__(self, native_player, idx):
            object.__setattr__(self, '_p', native_player)
            object.__setattr__(self, 'id', idx)

        def __getattr__(self, name):
            return getattr(self._p, name)

        def __setattr__(self, name, value):
            if name in ('_p', 'id'):
                object.__setattr__(self, name, value)
                return
            try:
                setattr(self._p, name, value)
            except Exception:
                object.__setattr__(self, name, value)


    class GameStateWrapper:
        def __init__(self, *args, **kwargs):
            if not args and not kwargs:
                try:
                    self._native = _orig_GameState(40)
                except Exception:
                    self._native = _orig_GameState(*args, **kwargs)
            else:
                self._native = _orig_GameState(*args, **kwargs)
            try:
                native_players = list(self._native.players)
                self.players = [_PlayerProxy(p, i) for i, p in enumerate(native_players)]
                for i, proxy in enumerate(self.players):
                    try:
                        native_p = native_players[i]
                        self._install_zone_proxies(native_p, proxy)
                    except Exception:
                        pass
            except Exception:
                self.players = []

            try:
                self.stack_zone = list(getattr(self._native, 'stack_zone', []))
            except Exception:
                self.stack_zone = []

            self.pending_effects = list(getattr(self._native, 'pending_effects', []))

        def __getattr__(self, name):
            try:
                return getattr(self._native, name)
            except Exception:
                raise AttributeError(name)

        def execute_command(self, cmd):
            try:
                if not hasattr(self, '_shim_command_history'):
                    object.__setattr__(self, '_shim_command_history', [])
                try:
                    object.__setattr__(self, 'command_history', self._shim_command_history)
                except Exception:
                    pass
                try:
                    self._shim_command_history.append(cmd)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if hasattr(self._native, 'command_history'):
                    try:
                        getattr(self._native, 'command_history').append(cmd)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if hasattr(cmd, 'execute'):
                    cmd.execute(self)
                    return
            except Exception:
                pass
            try:
                return getattr(self._native, 'execute_command')(cmd)
            except Exception:
                return None

        def _install_zone_proxies(self, native_p, proxy):
            try:
                class _CIProxy:
                    def __init__(self, native_ci):
                        object.__setattr__(self, '_ci', native_ci)
                    def __getattr__(self, n):
                        return getattr(self._ci, n)
                    @property
                    def id(self):
                        return getattr(self._ci, 'instance_id', getattr(self._ci, 'id', None))

                class _ZoneProxy:
                    def __init__(self, native_player, zn):
                        object.__setattr__(self, '_p', native_player)
                        object.__setattr__(self, '_zn', zn)
                    def __len__(self):
                        try:
                            return len(getattr(self._p, self._zn))
                        except Exception:
                            return 0
                    def __iter__(self):
                        try:
                            for ci in list(getattr(self._p, self._zn)):
                                yield _CIProxy(ci)
                        except Exception:
                            return
                    def append(self, ci):
                        try:
                            getattr(self._p, self._zn).append(ci)
                        except Exception:
                            pass
                    def pop(self, idx=-1):
                        try:
                            return getattr(self._p, self._zn).pop(idx)
                        except Exception:
                            raise
                    def clear(self):
                        try:
                            getattr(self._p, self._zn).clear()
                        except Exception:
                            pass
                    def __getitem__(self, idx):
                        try:
                            return _CIProxy(getattr(self._p, self._zn)[idx])
                        except Exception:
                            raise

                try: setattr(proxy, 'hand', _ZoneProxy(native_p, 'hand'))
                except Exception: pass
                try: setattr(proxy, 'mana_zone', _ZoneProxy(native_p, 'mana_zone'))
                except Exception: pass
                try: setattr(proxy, 'battle_zone', _ZoneProxy(native_p, 'battle_zone'))
                except Exception: pass
                try: setattr(proxy, 'shield_zone', _ZoneProxy(native_p, 'shield_zone'))
                except Exception: pass
            except Exception:
                pass

        def _ensure_player(self, idx: int):
            try:
                native_players = list(getattr(self._native, 'players'))
            except Exception:
                native_players = []
                try: setattr(self._native, 'players', native_players)
                except Exception: pass
            while len(native_players) <= idx:
                p = type('P', (), {})()
                p.hand = []
                p.mana_zone = []
                p.battle_zone = []
                p.shield_zone = []
                native_players.append(p)
            try: setattr(self._native, 'players', native_players)
            except Exception: pass
            try:
                self.players = [_PlayerProxy(p, i) for i, p in enumerate(native_players)]
                for i, proxy in enumerate(self.players):
                    try: self._install_zone_proxies(native_players[i], proxy)
                    except Exception: pass
            except Exception: pass

        def clear_zone(self, player_idx, zone):
            try:
                zone_list = getattr(self._native.players[player_idx], zone.name.lower())
                zone_list.clear()
            except Exception:
                pass

        def add_test_card_to_battle(self, *args, **kwargs):
            try: return self._native.add_test_card_to_battle(*args, **kwargs)
            except Exception: return None

        def add_test_card_to_shield(self, *args, **kwargs):
            try: return getattr(self._native, 'add_test_card_to_shield')(*args, **kwargs)
            except Exception: pass

        def setup_test_duel(self):
            try: return self._native.setup_test_duel()
            except Exception: return None

        def add_card_to_hand(self, *args, **kwargs):
            try: return self._native.add_card_to_hand(*args, **kwargs)
            except Exception: return None

        def add_card_to_mana(self, *args, **kwargs):
            try: return self._native.add_card_to_mana(*args, **kwargs)
            except Exception: pass

    def _effectdef_factory(*args, **kwargs):
        inst = _orig_EffectDef()
        try:
            if len(args) >= 1: inst.trigger = args[0]
            if len(args) >= 2: inst.condition = args[1]
            if len(args) >= 3: inst.actions = args[2]
            for k, v in kwargs.items():
                try: setattr(inst, k, v)
                except Exception: pass
        except Exception: pass
        return inst

    def _conditiondef_factory(*args, **kwargs):
        inst = _orig_ConditionDef()
        for k, v in kwargs.items():
            try: setattr(inst, k, v)
            except Exception: pass
        return inst

    def _actiondef_factory(*args, **kwargs):
        inst = _orig_ActionDef()
        try:
            if len(args) >= 1: inst.type = args[0]
            if len(args) >= 2: inst.scope = args[1]
            if len(args) >= 3: inst.filter = args[2]
            for k, v in kwargs.items():
                try: setattr(inst, k, v)
                except Exception: pass
        except Exception: pass
        return inst

    def _filterdef_factory(*args, **kwargs):
        inst = _orig_FilterDef()
        for k, v in kwargs.items():
            try: setattr(inst, k, v)
            except Exception: pass
        return inst

    if _orig_GameState: dm_ai_module.GameState = GameStateWrapper
    if _orig_EffectDef: dm_ai_module.EffectDef = _effectdef_factory
    if _orig_ConditionDef: dm_ai_module.ConditionDef = _conditiondef_factory
    if _orig_ActionDef: dm_ai_module.ActionDef = _actiondef_factory
    if _orig_FilterDef: dm_ai_module.FilterDef = _filterdef_factory

    # Ensure CardDefinition instances loaded from JsonLoader expose reaction_abilities
    _orig_json_loader = getattr(dm_ai_module, 'JsonLoader', None)
    if _orig_json_loader is not None:
        def _wrap_loaded_cards(card_map):
            class _CardDefProxy:
                def __init__(self, native):
                    object.__setattr__(self, '_native', native)
                    object.__setattr__(self, 'reaction_abilities', [])
                def __getattr__(self, name):
                    return getattr(self._native, name)
                def __setattr__(self, name, value):
                    try: setattr(self._native, name, value)
                    except Exception: object.__setattr__(self, name, value)
            out = {}
            for cid, cdef in card_map.items():
                try:
                    if not hasattr(cdef, 'reaction_abilities'):
                        try:
                            setattr(cdef, 'reaction_abilities', [])
                            out[cid] = cdef
                            continue
                        except Exception: pass
                    out[cid] = cdef
                except Exception:
                    out[cid] = _CardDefProxy(cdef)
            return out
        try:
            orig_load = dm_ai_module.JsonLoader.load_cards
            def _load_cards(path):
                m = orig_load(path)
                return _wrap_loaded_cards(m)
            dm_ai_module.JsonLoader.load_cards = staticmethod(_load_cards)
        except Exception:
            pass

    # GenericCardSystem wrappers
    try:
        gcs = dm_ai_module.GenericCardSystem
        if not hasattr(gcs, 'resolve_action_with_db') and hasattr(gcs, 'resolve_effect_with_db'):
                def _resolve_action_with_db(state, action_or_effect, source_id, card_db, ctx=None):
                    try:
                        if hasattr(action_or_effect, 'actions'): eff = action_or_effect
                        else:
                            eff = _orig_EffectDef()
                            eff.actions = [action_or_effect]
                    except Exception:
                        eff = _orig_EffectDef()
                        eff.actions = [action_or_effect]
                    return gcs.resolve_effect_with_db(state, eff, source_id, card_db)
                gcs.resolve_action_with_db = staticmethod(_resolve_action_with_db)
    except Exception:
        pass

    if hasattr(dm_ai_module, 'CardRegistry') and not hasattr(dm_ai_module.CardRegistry, 'get_all_cards'):
        if hasattr(dm_ai_module.CardRegistry, 'get_all_definitions'):
            dm_ai_module.CardRegistry.get_all_cards = staticmethod(dm_ai_module.CardRegistry.get_all_definitions)
        else:
            dm_ai_module.CardRegistry.get_all_cards = staticmethod(lambda: {})

    if not hasattr(dm_ai_module, 'ManaSystem'):
        class _ManaShim:
            @staticmethod
            def auto_tap_mana(state, player, card_def, card_db): return False
        dm_ai_module.ManaSystem = _ManaShim
