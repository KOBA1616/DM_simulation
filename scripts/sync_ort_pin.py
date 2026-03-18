#!/usr/bin/env python3
"""
Sync ONNX Runtime version pinned in CMakeLists.txt to requirements.txt.

Usage: python scripts/sync_ort_pin.py
"""
import re
from pathlib import Path
import sys


def extract_version_from_cmake(cmake_path: Path) -> str | None:
    text = cmake_path.read_text(encoding="utf-8")
    # Look for releases/download/vX.Y.Z pattern
    m = re.search(r"releases/download/v(\d+\.\d+\.\d+)", text)
    if m:
        return m.group(1)
    # Fallback: look for onnxruntime-...-X.Y.Z in URL or filename
    m2 = re.search(r"onnxruntime[-_][^/]*-(\d+\.\d+\.\d+)", text)
    if m2:
        return m2.group(1)
    return None


def update_requirements(req_path: Path, version: str) -> bool:
    text = req_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    out_lines = []
    for ln in lines:
        if re.match(r"^\s*onnxruntime(==|>=|<=|~=|\s|$)", ln):
            out_lines.append(f"onnxruntime=={version}")
            changed = True
        else:
            out_lines.append(ln)
    if changed:
        backup = req_path.with_suffix(req_path.suffix + ".cmake_sync.bak")
        req_path.replace(backup)
        # write new requirements
        req_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cmake = repo_root / "CMakeLists.txt"
    req = repo_root / "requirements.txt"
    if not cmake.exists():
        print("CMakeLists.txt not found", file=sys.stderr)
        return 2
    if not req.exists():
        print("requirements.txt not found", file=sys.stderr)
        return 2

    ver = extract_version_from_cmake(cmake)
    if not ver:
        print("Could not determine ONNX Runtime version from CMakeLists.txt", file=sys.stderr)
        return 3

    print(f"CMake ONNX Runtime version: {ver}")
    changed = update_requirements(req, ver)
    if changed:
        print(f"Updated {req} to pin onnxruntime=={ver}. Backup created.")
    else:
        print(f"No change needed; {req} already pins onnxruntime=={ver} (or no onnxruntime entry found).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
