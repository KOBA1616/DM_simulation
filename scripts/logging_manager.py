import logging
import logging.handlers
import sys
import os
from typing import Dict, Iterable, Optional
import time
import threading

# Internal runtime helpers for stronger enforcement
_CONFIGURED = False
_FH = None
_CH = None
_SILENT_PATTERNS = []
_ORIG_LOGGER_ADDHANDLER = None
_ORIG_BASICCONFIG = None

LOG_DEFAULT = 'tmp_selfplay_long.log'
DEFAULT_SILENT_LOGGERS = [
    # Noisy internal loggers that are rarely useful at INFO/DEBUG on console
    'EngineCompat',
    'dm_ai_module',
    'dm_toolkit',
]

# runtime config snapshot
_CURRENT_CONFIG: Dict[str, object] = {}


def configure_logging(
    log_file=LOG_DEFAULT,
    console_level_name=None,
    file_level_name=None,
    root_level_name=None,
    silent_loggers=None,
):
    """
    Central logging configuration API.

    Behavior:
    - Values passed explicitly override environment variables.
    - Environment variables (if present): DM_CONSOLE_LOG_LEVEL, DM_FILE_LOG_LEVEL,
      DM_ROOT_LOG_LEVEL, DM_SILENT_LOGGERS (comma-separated).
    """
    console_level_name = (
        console_level_name
        or os.getenv('DM_CONSOLE_LOG_LEVEL')
        or 'INFO'
    )
    file_level_name = (
        file_level_name
        or os.getenv('DM_FILE_LOG_LEVEL')
        or 'DEBUG'
    )
    root_level_name = (
        root_level_name
        or os.getenv('DM_ROOT_LOG_LEVEL')
        or 'DEBUG'
    )
    if silent_loggers is None:
        sl = os.getenv('DM_SILENT_LOGGERS')
        if sl is None:
            # use the built-in default noisy logger list when env not provided
            silent_loggers = list(DEFAULT_SILENT_LOGGERS)
        else:
            silent_loggers = [s.strip() for s in sl.split(',') if s.strip()]

    root = logging.getLogger()
    root.setLevel(getattr(logging, root_level_name.upper(), logging.DEBUG))

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # rotation params from env or defaults
    max_bytes = int(os.getenv('DM_LOG_MAX_BYTES', str(5 * 1024 * 1024)))
    backup_count = int(os.getenv('DM_LOG_BACKUP_COUNT', '3'))
    file_format = os.getenv('DM_LOG_FORMAT') or '%(asctime)s %(levelname)s: %(message)s'
    datefmt = os.getenv('DM_LOG_DATEFMT') or None
    formatter = logging.Formatter(file_format, datefmt=datefmt)

    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
        mode='a',
    )
    fh.setLevel(getattr(logging, file_level_name.upper(), logging.DEBUG))
    fh.setFormatter(formatter)

    # console handler: optionally send to stderr via env
    console_to_stderr = os.getenv('DM_LOG_CONSOLE_STDERR', '1') in ('1', 'true', 'True')
    ch_stream = sys.stderr if console_to_stderr else sys.stdout
    ch = logging.StreamHandler(stream=ch_stream)
    ch.setLevel(getattr(logging, console_level_name.upper(), logging.INFO))
    ch.setFormatter(formatter)

    # Optional simple rate-limiter for console to collapse very high-frequency messages
    # Controlled by env DM_LOG_RATE_LIMIT_SECONDS (float). 0 or unset disables.
    try:
        rate_limit_seconds = float(os.getenv('DM_LOG_RATE_LIMIT_SECONDS', '0'))
    except Exception:
        rate_limit_seconds = 0.0

    class RateLimitFilter(logging.Filter):
        def __init__(self, seconds: float = 0.0):
            super().__init__()
            self.seconds = float(seconds or 0.0)
            self._lock = threading.Lock()
            self._last: Dict[str, float] = {}

        def _normalize(self, text: str) -> str:
            # Remove memory addresses like 0x7ffdeadbeef and object reprs like
            # <module.Class object at 0x7ffdeadbeef>
            import re

            # strip common Python object reprs
            text = re.sub(r"<[^>]+ object at 0x[0-9A-Fa-f]+>", "<OBJ>", text)
            # strip bare hex addresses
            text = re.sub(r"0x[0-9A-Fa-f]+", "<HEX>", text)
            # strip UUIDs
            text = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<UUID>", text)
            # collapse long numeric sequences (optional)
            text = re.sub(r"\d{6,}", "<NUM>", text)
            return text

        def filter(self, record: logging.LogRecord) -> bool:
            if not self.seconds or self.seconds <= 0:
                return True
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            # Normalize message to group semantically-similar logs that only differ
            # by addresses/UIDs so rate-limiter can suppress repeats effectively.
            try:
                norm = self._normalize(msg)
            except Exception:
                norm = msg
            key = f"{record.name}:{norm}"
            now = time.time()
            with self._lock:
                last = self._last.get(key)
                if last is None or (now - last) >= self.seconds:
                    self._last[key] = now
                    return True
                return False

    if rate_limit_seconds and rate_limit_seconds > 0:
        ch.addFilter(RateLimitFilter(rate_limit_seconds))

    try:
        root.handlers.clear()
    except Exception:
        root.handlers = []
    root.addHandler(fh)
    root.addHandler(ch)

    # store created handlers for possible re-application by patched basicConfig
    global _FH, _CH, _SILENT_PATTERNS, _CONFIGURED
    _FH = fh
    _CH = ch
    _SILENT_PATTERNS = list(silent_loggers or [])

    # apply per-logger explicit levels if provided via env DM_LOGGER_LEVELS
    # format: "name=LEVEL,name2=LEVEL"
    logger_levels_env = os.getenv('DM_LOGGER_LEVELS', '')
    logger_levels: Dict[str, int] = {}
    if logger_levels_env:
        for part in logger_levels_env.split(','):
            if '=' in part:
                n, l = part.split('=', 1)
                n = n.strip()
                l = l.strip().upper()
                if n:
                    logger_levels[n] = getattr(logging, l, logging.NOTSET)

    for name in (silent_loggers or []):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass

    for name, level in logger_levels.items():
        try:
            logging.getLogger(name).setLevel(level)
        except Exception:
            pass

    # Ensure noisy loggers don't keep their own handlers that bypass root's console level.
    # For each logger already created in the logging manager, if it matches an entry
    # in `silent_loggers` (exact or prefix), remove its handlers and stop propagation
    # so its messages won't reach the console handler.
    try:
        existing = list(getattr(logging.root.manager, 'loggerDict', {}).items())
        for silent in (silent_loggers or []):
            for lname, lobj in existing:
                if not isinstance(lname, str):
                    continue
                # match exact name or prefix (e.g. 'dm_toolkit' matches 'dm_toolkit.sub')
                if lname == silent or lname.startswith(silent + '.') or lname.startswith(silent):
                    try:
                        if hasattr(lobj, 'handlers'):
                            lobj.handlers.clear()
                    except Exception:
                        try:
                            lobj.handlers = []
                        except Exception:
                            pass
                    try:
                        # stop propagation so root console handler controls output
                        if hasattr(lobj, 'propagate'):
                            lobj.propagate = False
                    except Exception:
                        pass
                    try:
                        lobj.setLevel(logging.WARNING)
                    except Exception:
                        pass
    except Exception:
        # best-effort only; don't fail logging configuration
        pass

    # Patch Logger.addHandler and logging.basicConfig to make later ad-hoc
    # handler additions a no-op for noisy loggers and to keep our handlers
    # authoritative. This is a best-effort guard against modules that call
    # `logger.addHandler` or `logging.basicConfig` after central config.
    global _ORIG_LOGGER_ADDHANDLER, _ORIG_BASICCONFIG
    if not _CONFIGURED:
        try:
            _ORIG_LOGGER_ADDHANDLER = getattr(logging.Logger, 'addHandler', None)

            def _patched_addHandler(self, hdlr):
                try:
                    name = getattr(self, 'name', '') or ''
                    # If this logger matches a silent pattern and the handler
                    # is a console-style StreamHandler, ignore the addition.
                    if any(name == p or name.startswith(p + '.') or name.startswith(p) for p in _SILENT_PATTERNS):
                        # Also ignore StreamHandler regardless of stream (defensive)
                        try:
                            if isinstance(hdlr, logging.StreamHandler):
                                # record diagnostic about ignored addHandler
                                try:
                                    import traceback, datetime
                                    diag = os.getenv('DM_LOG_ADDHANDLER_DIAG') or os.path.join(os.getcwd(), 'logging_addhandler_diag.log')
                                    with open(diag, 'a', encoding='utf-8') as df:
                                        df.write(f"\n--- addHandler ignored: {datetime.datetime.utcnow().isoformat()} UTC ---\n")
                                        df.write(f"logger={name}\nhandler={repr(hdlr)}\n")
                                        traceback.print_stack(file=df)
                                except Exception:
                                    pass
                                return
                        except Exception:
                            pass
                        # If non-stream handler added to a silent logger, ignore too
                        return
                except Exception:
                    pass
                if _ORIG_LOGGER_ADDHANDLER:
                    return _ORIG_LOGGER_ADDHANDLER(self, hdlr)

            logging.Logger.addHandler = _patched_addHandler
        except Exception:
            pass

        try:
            _ORIG_BASICCONFIG = getattr(logging, 'basicConfig', None)

            def _patched_basicConfig(*args, **kwargs):
                res = None
                if _ORIG_BASICCONFIG:
                    try:
                        res = _ORIG_BASICCONFIG(*args, **kwargs)
                    except Exception:
                        res = None
                # Remove any StreamHandler that basicConfig may have added to root
                try:
                    for h in list(root.handlers):
                        if isinstance(h, logging.StreamHandler) and h is not _CH:
                            try:
                                root.removeHandler(h)
                            except Exception:
                                pass
                    # Ensure our handlers are present
                    if _FH and _FH not in root.handlers:
                        root.addHandler(_FH)
                    if _CH and _CH not in root.handlers:
                        root.addHandler(_CH)
                except Exception:
                    pass
                return res

            logging.basicConfig = _patched_basicConfig
        except Exception:
            pass

        _CONFIGURED = True

    # snapshot config for callers
    _CURRENT_CONFIG.update({
        'log_file': log_file,
        'console_level': console_level_name,
        'file_level': file_level_name,
        'root_level': root_level_name,
        'silent_loggers': silent_loggers,
        'max_bytes': max_bytes,
        'backup_count': backup_count,
        'format': file_format,
        'datefmt': datefmt,
        'rate_limit_seconds': rate_limit_seconds,
    })

    return {
        'log_file': log_file,
        'console_level': console_level_name,
        'file_level': file_level_name,
        'root_level': root_level_name,
        'silent_loggers': silent_loggers,
    }


def quiet_loggers(names: Iterable[str]) -> None:
    """Programmatically set a list of logger names to WARNING level.

    Useful for callers that want to silence noisy subsystems at runtime.
    """
    for n in names:
        try:
            logging.getLogger(n).setLevel(logging.WARNING)
        except Exception:
            pass


def get_logger(name, level=None):
    """Return a logger configured to inherit handlers from the root.

    If `level` is provided, set the logger's level explicitly.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.NOTSET))
    return logger


def set_logger_level(name: str, level_name: str) -> None:
    """Set a specific logger's level at runtime."""
    try:
        lvl = getattr(logging, level_name.upper(), None)
        if lvl is None:
            return
        logging.getLogger(name).setLevel(lvl)
    except Exception:
        pass


def get_config() -> Dict[str, object]:
    """Return the last applied logging configuration."""
    return dict(_CURRENT_CONFIG)
