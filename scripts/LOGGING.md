# Logging Manager (scripts/logging_manager.py)

This project uses a centralized logging manager at `scripts/logging_manager.py`.
Use `configure_logging(...)` from entrypoints to ensure consistent console/file handlers and rotation.

Environment variables (defaults set by `scripts/run_gui.ps1` for GUI runs):
- `DM_CONSOLE_LOG_LEVEL` — Console handler level (e.g. `WARNING`).
- `DM_FILE_LOG_LEVEL` — File handler level (e.g. `DEBUG`).
- `DM_ROOT_LOG_LEVEL` — Root logger level.
- `DM_SILENT_LOGGERS` — Comma-separated logger names to silence (set to `EngineCompat,dm_ai_module,dm_toolkit` by default for GUI runs).
- `DM_LOG_RATE_LIMIT_SECONDS` — Float seconds to rate-limit semantically-similar console messages (0 disables).
- `DM_LOG_MAX_BYTES` — Rotating file max size in bytes.
- `DM_LOG_BACKUP_COUNT` — Rotating file backup count.
- `DM_LOG_FORMAT` — Python logging `Formatter` format string used for both handlers unless overridden.
- `DM_LOG_DATEFMT` — Optional datefmt for the formatter.
- `DM_LOG_CONSOLE_STDERR` — If `1` (default), console output writes to `stderr`; otherwise `stdout`.
- `DM_LOGGER_LEVELS` — Comma-separated `name=LEVEL` pairs to set specific logger levels (e.g. `EngineCompat=WARNING,dm_ai_module=ERROR`).

Programmatic helpers (from `scripts.logging_manager`):
- `configure_logging(log_file, console_level_name, file_level_name, root_level_name, silent_loggers)` — central configuration; explicit args override env.
- `get_logger(name, level=None)` — returns a logger; optionally set its level.
- `set_logger_level(name, level_name)` — set a logger level at runtime.
- `quiet_loggers(names)` — convenience to set multiple logger names to `WARNING`.
- `get_config()` — returns last applied config snapshot.

Quick usage examples:

Run GUI with quieter console and full file debug:

```powershell
$env:DM_CONSOLE_LOG_LEVEL = 'WARNING'
$env:DM_FILE_LOG_LEVEL = 'DEBUG'
& scripts/run_gui.ps1
```

Programmatic (inside a script):

```python
from scripts.logging_manager import configure_logging, get_logger
configure_logging(log_file='myrun.log', console_level_name='INFO', file_level_name='DEBUG')
log = get_logger(__name__)
log.info('started')
```

Notes:
- The manager includes a simple console rate-limiter that normalizes messages (strips object addresses and UUIDs) to better collapse repeated debug lines. If logs still appear noisy, prefer raising `DM_CONSOLE_LOG_LEVEL` or silencing specific noisy loggers via `DM_SILENT_LOGGERS` or `DM_LOGGER_LEVELS`.
- Third-party packages and your virtualenv may call `logging.basicConfig()`; the repo's own code should use `configure_logging` instead. The repo scan excluded virtualenv packages.
