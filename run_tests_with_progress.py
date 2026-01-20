#!/usr/bin/env python3
"""Run pytest and show a concise progress / completion percentage.

Usage:
  python run_tests_with_progress.py [pytest args]

If no args are given, runs `pytest -q` in the repository.
"""
import sys
import time
import pytest


class ProgressPlugin:
    def __init__(self):
        self.total = None
        self.finished = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.start = time.time()

    def pytest_collection_modifyitems(self, session, config, items):
        # Called after collection; set total
        self.total = len(items)
        if self.total == 0:
            print('No tests collected')
        else:
            print(f'Total tests: {self.total}')

    def pytest_runtest_logreport(self, report):
        # report.when in ('setup','call','teardown') - count on 'call' to reflect outcome
        if report.when != 'call':
            return
        self.finished += 1
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1

        # compute percent
        total = self.total or 1
        pct = (self.finished / total) * 100
        # concise single-line progress
        bar_len = 30
        filled = int((self.finished / total) * bar_len)
        bar = '[' + ('=' * filled).ljust(bar_len) + ']'
        msg = f"{bar} {self.finished}/{total} ({pct:5.1f}%)  passed={self.passed} failed={self.failed} skipped={self.skipped}"
        # overwrite same line when possible
        sys.stdout.write('\r' + msg)
        sys.stdout.flush()

    def pytest_sessionfinish(self, session, exitstatus):
        # Newline then summary
        try:
            sys.stdout.write('\n')
            dur = time.time() - self.start
            print(f"Finished: passed={self.passed} failed={self.failed} skipped={self.skipped} total={self.total or 0} time={dur:.2f}s")
        except Exception:
            pass


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        pytest_args = ['-q']
    else:
        pytest_args = argv

    plugin = ProgressPlugin()
    # Run pytest with our plugin
    return pytest.main(pytest_args, plugins=[plugin])


if __name__ == '__main__':
    raise SystemExit(main())
