#!/usr/bin/env python3
"""繰り返し pytest を実行して失敗を再現するためのハーネス。

使い方: python scripts/repeat_card1_hand_quality.py --count 50
出力:
 - logs/repro_card1/run_{i}.log
 - logs/repro_card1/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_once(pytest_target: str, log_path: str, kfilter: str | None = None) -> dict:
    cmd = [sys.executable, "-m", "pytest", pytest_target, "-q", "-rA"]
    if kfilter:
        cmd.extend(["-k", kfilter])
    started = datetime.utcnow().isoformat() + "Z"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ended = datetime.utcnow().isoformat() + "Z"
    out = proc.stdout.decode(errors="replace")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n")
        f.write(f"# START: {started}\n# END: {ended}\n\n")
        f.write(out)
    return {
        "returncode": proc.returncode,
        "log": log_path,
        "start": started,
        "end": ended,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "-n", type=int, default=20)
    parser.add_argument("--target", "-t", default="tests/test_card1_hand_quality.py", help="pytest target (file or nodeid). Use -k to filter test names inside classes")
    parser.add_argument("--k", "-k", dest="kfilter", default=None, help="pytest -k expression to filter test names")
    parser.add_argument("--log-dir", "-d", default="logs/repro_card1")
    args = parser.parse_args()

    ensure_dir(args.log_dir)

    summary = {
        "target": args.target,
        "count": args.count,
        "runs": [],
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    failures = 0
    for i in range(1, args.count + 1):
        log_path = os.path.join(args.log_dir, f"run_{i}.log")
        res = run_once(args.target, log_path, kfilter=args.kfilter)
        ok = res["returncode"] == 0
        summary["runs"].append({"i": i, "ok": ok, **res})
        if not ok:
            failures += 1
            # on first failure, capture a small copy of the failing log as last_failure.log
            try:
                with open(log_path, "r", encoding="utf-8") as fin:
                    excerpt = fin.read(10 * 1024)
                with open(os.path.join(args.log_dir, "last_failure.log"), "w", encoding="utf-8") as fout:
                    fout.write(excerpt)
            except Exception:
                pass

    summary["finished_at"] = datetime.utcnow().isoformat() + "Z"
    summary["failures"] = failures
    summary_path = os.path.join(args.log_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Runs: {args.count}, Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
