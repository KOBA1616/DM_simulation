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

        # Force-load the repository dm_ai_module.py wrapper into sys.modules so
        # tests use the Python stub/wrapper instead of accidentally importing a
        # partial/native extension that may not expose the expected enums.
        try:
            import sys, os, importlib.util
            root = os.path.dirname(os.path.abspath(__file__))
            wrapper_path = os.path.join(root, 'dm_ai_module.py')
            if os.path.exists(wrapper_path):
                spec = importlib.util.spec_from_file_location('dm_ai_module', wrapper_path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules['dm_ai_module'] = module
                    if spec.loader is not None:
                        spec.loader.exec_module(module)
        except Exception:
            # Best-effort only; don't fail test startup on platform-specific issues
            pass

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
