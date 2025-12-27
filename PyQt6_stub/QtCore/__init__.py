class _Signal:
    def __init__(self):
        self._callbacks = []
    def connect(self, cb):
        try:
            self._callbacks.append(cb)
        except Exception:
            pass
    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except Exception:
            pass
    def emit(self, *args, **kwargs):
        for cb in list(self._callbacks):
            try:
                cb(*args, **kwargs)
            except Exception:
                pass


def pyqtSignal(*args, **kwargs):
    return _Signal()


class QTimer:
    def __init__(self):
        self.timeout = _Signal()
    def start(self, ms=0):
        pass
    def stop(self):
        pass


class Qt:
    class DockWidgetArea:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2
        AllDockWidgetAreas = 3
    class ItemDataRole:
        UserRole = 256
        DisplayRole = 0
    class AlignmentFlag:
        AlignLeft = 0
        AlignCenter = 1
        AlignRight = 2
        AlignTop = 4
        AlignBottom = 8
    class ScrollBarPolicy:
        ScrollBarAlwaysOff = 0
        ScrollBarAlwaysOn = 1
    class Orientation:
        Horizontal = 0
        Vertical = 1
    class CursorShape:
        PointingHandCursor = 1


class QSize:
    def __init__(self, w=0, h=0):
        self.width = w
        self.height = h


class QRect:
    def __init__(self, *args):
        pass


class QRectF:
    def __init__(self, *args):
        pass


class QThread:
    def __init__(self):
        pass
    def start(self):
        pass
    def quit(self):
        pass


class QMimeData:
    def __init__(self):
        pass


class QModelIndex:
    def __init__(self):
        pass


__all__ = ["QTimer", "Qt", "pyqtSignal", "QSize", "QRect", "QRectF", "QThread", "QMimeData", "QModelIndex"]
