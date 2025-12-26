import os
import json
import time
from typing import Optional

_ENABLED = os.environ.get('DM_MIGRATION_TELEMETRY', '0') == '1'
_METRICS_PATH = os.environ.get('DM_MIGRATION_METRICS_PATH', os.path.join(os.getcwd(), 'migration_metrics.jsonl'))


def _open_append(path: str):
    return open(path, 'a', encoding='utf-8')


def record_conversion(success: bool, action_type: Optional[str] = None, warning: Optional[str] = None) -> None:
    """Record a single conversion event as a JSON line if telemetry is enabled."""
    if not _ENABLED:
        return
    entry = {
        'ts': time.time(),
        'success': bool(success),
        'action_type': action_type,
        'warning': warning,
    }
    try:
        with _open_append(_METRICS_PATH) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        # Silently ignore telemetry failures to avoid disrupting editor
        pass


def read_metrics(limit: int = 1000):
    """Return last N metrics entries (best-effort)."""
    if not os.path.exists(_METRICS_PATH):
        return []
    out = []
    try:
        with open(_METRICS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out[-limit:]
