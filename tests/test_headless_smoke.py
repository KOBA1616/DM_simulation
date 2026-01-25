import sys
import subprocess


def test_headless_smoke_runs():
    """Run the existing smoke script and assert it completes and reports legal commands."""
    proc = subprocess.run([sys.executable, "scripts/headless_smoke.py"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    out = proc.stdout.decode(errors="replace")
    assert proc.returncode == 0, f"smoke script failed:\n{out}"
    assert "legal commands" in out or "legal commands total" in out, "expected legal-commands text not found in output"