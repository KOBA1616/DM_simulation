"""Update docs/systems/native_bridge/missing_native_symbols.md from scripts/list_dm_symbols.py output.

Behavior:
- Runs scripts/list_dm_symbols.py and parses JSON list of exported symbols.
- If import fails, attempts to locate local `.pyd` build artifacts and load them directly as a fallback.
- If all attempts fail, appends a diagnostic note to the docs so the operator can debug.
"""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'scripts' / 'list_dm_symbols.py'
MD = ROOT / 'docs' / 'systems' / 'native_bridge' / 'missing_native_symbols.md'

if not SCRIPT.exists():
    print('ERROR: list_dm_symbols.py not found', file=sys.stderr)
    sys.exit(2)

def try_run_list_script() -> tuple[int, str, str]:
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def try_load_pyd_candidates() -> tuple[bool, list, list]:
    """Search common locations for dm_ai_module*.pyd and try to load them directly.

    Returns (success, symbols_list, diagnostics).
    """
    candidates = []
    candidates += list((ROOT / 'bin').glob('dm_ai_module*.pyd'))
    candidates += list((ROOT / 'build').rglob('dm_ai_module*.pyd'))
    candidates += list((ROOT / 'build-ninja').rglob('dm_ai_module*.pyd'))
    diagnostics = []
    for p in candidates:
        try:
            spec = importlib.util.spec_from_file_location('dm_ai_module', str(p))
            if spec is None or spec.loader is None:
                diagnostics.append(f'bad spec for {p}')
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            syms = sorted([x for x in dir(module) if not x.startswith('_')])
            return True, syms, [f'Loaded from {p}']
        except Exception as e:
            diagnostics.append(f'Failed to load {p}: {e}')
    return False, [], diagnostics


rc, out, err = try_run_list_script()

syms = None
diag_msgs = []

if rc == 0:
    try:
        syms = json.loads(out)
    except Exception as e:
        print('ERROR parsing output as JSON:', e, file=sys.stderr)
        sys.exit(3)
else:
    diag_msgs.append(f'list_dm_symbols failed: returncode={rc} out={out} err={err}')
    ok, syms_try, diagnostics = try_load_pyd_candidates()
    diag_msgs.extend(diagnostics)
    if ok:
        syms = syms_try
    else:
        note = f"\n\n> Automatic audit attempted on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}, but list extraction failed.\n> Diagnostics:\n> " + '\n> '.join(diag_msgs)
        print('IMPORT FAILED (and pyd fallbacks failed), writing note to doc')
        text = MD.read_text(encoding='utf-8')
        text += '\n\n' + '### Last automated audit result\n' + note + '\n'
        MD.write_text(text, encoding='utf-8')
        sys.exit(1)

# build new Present section
present_lines = []
present_lines.append('# Native symbols report (auto-generated snapshot)\n')
present_lines.append('\n')
present_lines.append(f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}')
present_lines.append('\n\n')
present_lines.append('This file records the symbols currently exported by the native `dm_ai_module` Python extension and highlights the small set of previously-expected symbols that remain unexported.\n\n')

present_lines.append('## Present (exported symbols)\n\n')
# write symbols sorted and grouped into columns (one per line)
for s in syms:
    present_lines.append(s)

present_text = '\n'.join(present_lines) + '\n\n'

# read existing file and preserve the "Missing" section if present
orig = MD.read_text(encoding='utf-8')
if '## Missing (previously expected but NOT exported under simple names)' in orig:
    parts = orig.split('## Missing (previously expected but NOT exported under simple names)')
    missing_part = '## Missing (previously expected but NOT exported under simple names)' + parts[1]
else:
    missing_part = '\n'

new_text = present_text + missing_part
MD.write_text(new_text, encoding='utf-8')
print(f'Updated {MD} with {len(syms)} symbols')
sys.exit(0)
