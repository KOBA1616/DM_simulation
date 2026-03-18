def pytest_configure(config):
    """After options are parsed, convert any quiet counts to verbose counts.

    This ensures running `pytest -q` results in verbose output for this repo.
    """
    # `quiet` and `verbose` are integer counts for -q/-v occurrences.
    quiet = getattr(config.option, "quiet", 0)
    if quiet:
        current_verbose = getattr(config.option, "verbose", 0) or 0
        # make verbose at least as many as quiet (e.g. -qq -> -vv)
        config.option.verbose = max(current_verbose, quiet)
        # disable quiet so pytest doesn't suppress output
        config.option.quiet = 0
        # NOTE: 再発防止 — フォールバック (dm_ai_module.py) は削除済み。
        # ネイティブ .pyd が利用可能な状態では -q フラグでも native を使う。

class _ProgressPlugin:
    """Simple pytest plugin to print a one-line progress bar and counts."""
    def __init__(self):
        self.total = 0
        self.finished = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_collection_modifyitems(self, session, config, items):
        self.total = len(items)
        if self.total:
            print(f"Total tests: {self.total}")

    def pytest_runtest_logreport(self, report):
        if report.when != 'call':
            return
        self.finished += 1
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1

        total = self.total or 1
        pct = (self.finished / total) * 100
        bar_len = 30
        filled = int((self.finished / total) * bar_len)
        bar = '[' + ('=' * filled).ljust(bar_len) + ']'
        msg = f"{bar} {self.finished}/{total} ({pct:5.1f}%) passed={self.passed} failed={self.failed} skipped={self.skipped}"
        try:
            # overwrite single line when terminal supports carriage return
            print('\r' + msg, end='')
        except Exception:
            print(msg)


def pytest_addoption(parser):
    # register plugin only when running tests (avoid interfering with other tooling)
    parser.addini('enable_progress_plugin', 'Enable progress bar plugin', default='true')


def pytest_cmdline_main(config):
    # Install plugin if enabled
    try:
        enabled = config.getini('enable_progress_plugin')
        if str(enabled).lower() in ('1', 'true', 'yes', 'on'):
            plugin = _ProgressPlugin()
            config.pluginmanager.register(plugin, name='progress-plugin')
    except Exception:
        pass


# Ensure a QApplication exists for tests that require Qt widgets. This avoids
# order-dependent failures where a test tries to construct a QWidget before
# a QApplication is created (observed during shuffled test runs).
import sys
from pytest import fixture


@fixture(scope='session', autouse=True)
def ensure_qt_app():
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception:
        try:
            from PySide2.QtWidgets import QApplication
        except Exception:
            # Qt not available in this environment; nothing to do.
            # 再発防止: generator fixture で return すると
            # "did not yield a value" が発生するため必ず yield する。
            yield
            return

    if QApplication.instance() is None:
        app = QApplication(sys.argv[:])
        yield
        try:
            app.quit()
        except Exception:
            pass
    else:
        yield
