#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
BUILD_REPORT = REPORTS / 'build' / 'quick_build_stdout.txt'
TEST_REPORT = REPORTS / 'tests' / 'pytest_latest.txt'

def ensure_dirs():
    (REPORTS / 'build').mkdir(parents=True, exist_ok=True)
    (REPORTS / 'tests').mkdir(parents=True, exist_ok=True)

def run_build():
    cmd = ['pwsh', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts' / 'quick_build.ps1')]
    print('Running build:', ' '.join(cmd))
    with open(BUILD_REPORT, 'wb') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    return proc.returncode

def run_tests():
    # Run pytest and save output
    cmd = ['pwsh', '-NoProfile', '-Command', f". {str(ROOT / '.venv' / 'Scripts' / 'Activate.ps1')} ; pytest tests/ -q --tb=short"]
    print('Running tests via PowerShell; ensure .venv exists and matches build Python')
    with open(TEST_REPORT, 'wb') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, shell=False)
    return proc.returncode

def run_enum():
    # Run the actiondef refs enumerator and capture output to reports/actiondef_refs.txt
    enum_script = ROOT / 'scripts' / 'list_actiondef_refs.py'
    enum_report = REPORTS / 'actiondef_refs.txt'
    if not enum_script.exists():
        print('Enumerator script not found:', enum_script)
        return 0
    cmd = [str(Path(os.environ.get('PYTHON', 'python'))), str(enum_script)]
    print('Running enumerator:', ' '.join(cmd))
    with open(enum_report, 'wb') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    return proc.returncode

def main():
    ensure_dirs()
    rc_build = run_build()
    print('Build exit code:', rc_build)
    rc_tests = run_tests()
    print('Tests exit code:', rc_tests)
    rc_enum = run_enum()
    print('Enum exit code:', rc_enum)
    if rc_build != 0 or rc_tests != 0:
        raise SystemExit(max(rc_build, rc_tests))

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Run quick_build.ps1 (Windows PowerShell) then run pytest and collect reports.

Usage: python scripts/run_build_and_tests.py

Notes:
- Must be run on Windows with PowerShell/PowerShell Core available in PATH as 'pwsh' or 'powershell'.
- Requires a working Python virtualenv matching the build Python ABI if native extension is built.
- This script only automates invocation and report collection; it does not modify source.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
REPORTS_BUILD = ROOT / 'reports' / 'build'
REPORTS_TESTS = ROOT / 'reports' / 'tests'

def run_powershell_script(script_path: Path, args=None, timeout=1800):
    args = args or []
    # Prefer pwsh (PowerShell Core) if available
    for shell in ('pwsh', 'powershell'):
        try:
            p = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(script_path)] + args,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return p.returncode, p.stdout
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return 124, 'TIMEOUT'
    return 127, 'No PowerShell (pwsh/powershell) found in PATH'


def main():
    print('Running quick_build.ps1...')
    build_script = SCRIPTS / 'quick_build.ps1'
    if not build_script.exists():
        print('quick_build.ps1 not found at', build_script)
        sys.exit(2)

    rc, out = run_powershell_script(build_script)
    print(out)

    # Ensure reports directories
    REPORTS_BUILD.mkdir(parents=True, exist_ok=True)
    REPORTS_TESTS.mkdir(parents=True, exist_ok=True)

    # Save build output
    build_out_file = REPORTS_BUILD / 'quick_build_stdout.txt'
    with open(build_out_file, 'w', encoding='utf8') as f:
        f.write(out or '')

    # Check for pyd
    pyd = ROOT / 'bin' / 'dm_ai_module.cp312-win_amd64.pyd'
    if pyd.exists():
        print('[BUILD] PYD exists at', pyd)
        try:
            shutil.copy2(pyd, ROOT / pyd.name)
            print('Copied PYD to repo root.')
        except Exception as e:
            print('Warning: failed to copy pyd to root:', e)
    else:
        print('[BUILD] PYD not found; build failed or output path differs.')

    # Run pytest
    print('Running pytest...')
    try:
        p = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=short'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        test_out = p.stdout
        test_rc = p.returncode
    except Exception as e:
        print('Failed to run pytest:', e)
        sys.exit(3)

    test_report_file = REPORTS_TESTS / 'pytest_latest.txt'
    with open(test_report_file, 'w', encoding='utf8') as f:
        f.write(test_out or '')

    print(test_out)
    print('Build return code:', rc)
    print('Pytest return code:', test_rc)

    # Exit with pytest rc if build succeeded (pyd exists), otherwise with build rc
    if pyd.exists():
        sys.exit(test_rc)
    else:
        sys.exit(rc)

if __name__ == '__main__':
    main()
