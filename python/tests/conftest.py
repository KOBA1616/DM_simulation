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
    
    # Create dummy classes
    class DummyQWidget:
        def __init__(self, *args, **kwargs): pass
        def setWindowTitle(self, title): pass
        def setLayout(self, layout): pass
        def show(self): pass

    class DummyQMainWindow(DummyQWidget):
        def setCentralWidget(self, widget): pass

    class DummyQt:
        class ItemDataRole:
            DisplayRole = 0
        Horizontal = 1

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
    for name in ['QLabel', 'QPushButton', 'QVBoxLayout', 'QHBoxLayout', 'QTreeWidget', 
                 'QTreeWidgetItem', 'QDialog', 'QLineEdit', 'QTextEdit', 'QCheckBox',
                 'QComboBox', 'QScrollArea', 'QTabWidget', 'QDockWidget', 'QGraphicsView',
                 'QGraphicsScene', 'QGraphicsEllipseItem', 'QGraphicsLineItem', 'QGraphicsTextItem',
                 'QProgressBar', 'QHeaderView', 'QSplitter', 'QGroupBox', 'QMenuBar', 'QMenu',
                 'QStatusBar']:
        setattr(qt_widgets, name, type(name, (DummyQWidget,), {}))

    qt_core.Qt = DummyQt
    qt_core.QObject = type('QObject', (), {'__init__': lambda s, *a: None})
    qt_core.QTimer = type('QTimer', (), {'singleShot': lambda *a: None})
    qt_core.pyqtSignal = lambda *args: unittest.mock.MagicMock(emit=lambda *a: None, connect=lambda *a: None)
    qt_core.QMimeData = type('QMimeData', (), {})
    qt_core.QRectF = type('QRectF', (), {})
    qt_core.QThread = type('QThread', (), {})

    qt_gui.QAction = type('QAction', (DummyQWidget,), {})
    qt_gui.QKeySequence = type('QKeySequence', (), {})
    qt_gui.QStandardItem = type('QStandardItem', (), {'__init__': lambda s, *a: None})
    qt_gui.QDrag = type('QDrag', (), {})
    qt_gui.QPen = type('QPen', (), {})
    qt_gui.QBrush = type('QBrush', (), {})
    qt_gui.QColor = type('QColor', (), {})
    qt_gui.QFont = type('QFont', (), {})
    qt_gui.QPainter = type('QPainter', (), {})
    qt_gui.QIcon = lambda *a: None
    qt_gui.QStandardItemModel = type('QStandardItemModel', (), {'__init__': lambda s, *a: None})

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
