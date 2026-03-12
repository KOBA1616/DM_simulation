import json
import sys
import subprocess
from pathlib import Path

def test_update_missing_native_symbols_updates_md(tmp_path):
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    doc_dir = repo_root / "docs" / "systems" / "native_bridge"
    scripts_dir.mkdir(parents=True)
    doc_dir.mkdir(parents=True)

    # Write a minimal list_dm_symbols.py that outputs JSON
    list_py = scripts_dir / "list_dm_symbols.py"
    list_py.write_text('import json; print(json.dumps(["symA","symB"]))')

    # Copy the real update script into the temp scripts dir
    orig_update = Path(__file__).resolve().parents[1] / 'scripts' / 'update_missing_native_symbols.py'
    assert orig_update.exists(), "update_missing_native_symbols.py not found in repo"
    upd_code = orig_update.read_text(encoding='utf-8')
    (scripts_dir / 'update_missing_native_symbols.py').write_text(upd_code, encoding='utf-8')

    # Create an initial missing_native_symbols.md with a Missing section to ensure preservation
    md_file = doc_dir / 'missing_native_symbols.md'
    md_file.write_text('# Native symbols report\n\n## Missing (previously expected but NOT exported under simple names)\n\n- old_sym\n')

    # Run the temp update script
    proc = subprocess.run([sys.executable, str(scripts_dir / 'update_missing_native_symbols.py')], cwd=str(repo_root), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"

    # Read updated MD and assert it contains Present section and our symbols
    updated = md_file.read_text(encoding='utf-8')
    assert '## Present (exported symbols)' in updated
    assert 'symA' in updated and 'symB' in updated
