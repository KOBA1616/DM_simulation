import glob
import subprocess
import sys
from pathlib import Path

def run_py(files, timeout=120):
    cmd = [sys.executable, "-m", "pytest", "-q"] + files
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout, False
    except subprocess.TimeoutExpired as e:
        return None, e.stdout or "", True


def collect_tests_in_file(f):
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", f]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = p.stdout.splitlines()
    tests = [l.strip() for l in lines if l.strip()]
    return tests


def find_hang(files):
    files = sorted(files)
    left = 0
    right = len(files)
    # If overall suite quick check
    print(f"Total test files: {len(files)}")
    code, out, timed = run_py(files, timeout=120)
    if timed:
        print("Entire suite timed out — proceeding to bisection")
    elif code != 0:
        print("Entire suite returned non-zero (failures). We'll still bisect to find problematic file(s).")
    else:
        print("Entire suite passed quickly (unexpected).")
        return

    # Binary search for problematic file
    l = 0
    r = len(files)
    candidate = None
    while l < r:
        mid = (l + r) // 2
        group = files[l:mid+1]
        print(f"Testing group {l}:{mid} -> {len(group)} files")
        code, out, timed = run_py(group, timeout=120)
        if timed or (code is not None and code != 0):
            # problem in left group
            r = mid
            candidate_group = group
            candidate = (l, mid)
        else:
            l = mid + 1
    if candidate is None:
        print("Could not find a single problematic group — consider increasing timeout or running more diagnostics.")
        return
    # Narrow to file by running each file in candidate_group
    print(f"Candidate group files: {candidate_group}")
    for f in candidate_group:
        print(f"Testing file: {f}")
        code, out, timed = run_py([f], timeout=120)
        if timed:
            print(f"File {f} times out (hang).")
            suspect = f
            break
        elif code != 0:
            print(f"File {f} has failures (non-zero).")
            suspect = f
            break
    else:
        print("No single file in candidate group failed or timed out.")
        return

    print(f"Suspect file: {suspect}")
    # Collect tests in suspect file and test individually
    tests = collect_tests_in_file(suspect)
    print(f"Collected tests in {suspect}: {tests}")
    for t in tests:
        # t may be like tests/ai/test_heuristic.py::test_heuristic
        if "::" not in t:
            node = f"{suspect}::{t}"
        else:
            node = t
        print(f"Running {node}")
        code, out, timed = run_py([node], timeout=120)
        if timed:
            print(f"Test {node} timed out (hang).")
            print(out)
            return
        elif code != 0:
            print(f"Test {node} failed (non-zero). Output:\n{out}")
            # continue to find hangs
        else:
            print(f"Test {node} passed.")


if __name__ == '__main__':
    base = Path('tests')
    files = [str(p) for p in base.rglob('*.py') if '__pycache__' not in str(p) and p.name != 'conftest.py']
    if not files:
        print('No test files found')
        sys.exit(1)
    find_hang(files)
