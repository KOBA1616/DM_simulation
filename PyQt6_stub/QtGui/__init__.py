class QAction:
    def __init__(self, text=None, parent=None):
        self._text = text
        self._parent = parent
        self.triggered = type("_S", (), {"connect": lambda self, cb: None})()
    def setToolTip(self, txt):
        pass
    def setCheckable(self, flag):
        pass

class QStandardItem:
    def __init__(self, *args, **kwargs):
        pass

class QStandardItemModel:
    def __init__(self, *args, **kwargs):
        pass

class QKeySequence:
    def __init__(self, *args, **kwargs):
        pass

class QFont:
    def __init__(self, *args, **kwargs):
        self._bold = False
        self._point_size = None
    def setBold(self, flag: bool):
        self._bold = bool(flag)
    def setPointSize(self, size: int):
        try:
            self._point_size = int(size)
        except Exception:
            self._point_size = None

class QColor:
    def __init__(self, *args, **kwargs):
        pass

class QCursor:
    pass

class QPainter:
    class RenderHint:
        Antialiasing = 1
    def __init__(self, *args, **kwargs):
        pass

class QPen:
    pass

class QBrush:
    pass

class QIcon:
    def __init__(self, *args, **kwargs):
        pass

def _make_dummy(name):
    class _D:
        def __init__(self, *a, **k):
            pass
    _D.__name__ = name
    return _D

_common = [
    'QKeySequence','QFont','QColor','QCursor','QPainter','QPen','QBrush','QIcon','QDrag'
]
for _n in _common:
    globals().setdefault(_n, _make_dummy(_n))

def __getattr__(name: str):
    if name.startswith('_'):
        raise AttributeError(name)
    val = globals().get(name)
    if val is not None:
        return val
    dummy = _make_dummy(name)
    globals()[name] = dummy
    return dummy

__all__ = ["QAction", "QStandardItem", "QStandardItemModel"]
class QAction:
    def __init__(self, text=None, parent=None):
        self._text = text
        self._parent = parent
        self.triggered = type("_S", (), {"connect": lambda self, cb: None})()
    def setToolTip(self, txt):
        pass
    def setCheckable(self, flag):
        pass

class QStandardItem:
    def __init__(self, *args, **kwargs):
        pass

class QStandardItemModel:
    def __init__(self, *args, **kwargs):
        pass

__all__ = ["QAction", "QStandardItem", "QStandardItemModel"]
