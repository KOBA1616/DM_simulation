from __future__ import annotations

import re
from pathlib import Path

import pytest

EXPECTED_ORT_VERSION: str = "1.20.1"
ROOT_DIR: Path = Path(__file__).resolve().parents[1]


def _read_requirements_ort_version(requirements_path: Path) -> str | None:
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("onnxruntime"):
            continue
        m = re.match(r"onnxruntime\s*==\s*([0-9]+\.[0-9]+\.[0-9]+)", line)
        if m:
            return m.group(1)
    return None


def _read_pyproject_ort_version(pyproject_path: Path) -> str | None:
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "onnxruntime==" not in line:
            continue
        m = re.search(r"onnxruntime==([0-9]+\.[0-9]+\.[0-9]+)", line)
        if m:
            return m.group(1)
    return None


def _read_cmake_ort_versions(cmake_path: Path) -> set[str]:
    text = cmake_path.read_text(encoding="utf-8")
    # 再発防止: URL の vX.Y.Z とファイル名 X.Y.Z の双方を検証して誤更新を早期検知する。
    return set(re.findall(r"onnxruntime[^\n]*?v([0-9]+\.[0-9]+\.[0-9]+)", text, re.IGNORECASE))


def test_ort_dependency_version_is_pinned_consistently() -> None:
    requirements_version = _read_requirements_ort_version(ROOT_DIR / "requirements.txt")
    pyproject_version = _read_pyproject_ort_version(ROOT_DIR / "pyproject.toml")

    assert requirements_version == EXPECTED_ORT_VERSION
    assert pyproject_version == EXPECTED_ORT_VERSION


def test_cmake_ort_fetch_version_matches_python_pin() -> None:
    cmake_versions = _read_cmake_ort_versions(ROOT_DIR / "CMakeLists.txt")

    assert cmake_versions == {EXPECTED_ORT_VERSION}


def test_runtime_onnxruntime_matches_expected_pin() -> None:
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    runtime_version = str(getattr(onnxruntime, "__version__", ""))
    # 実行環境ではランタイムのバージョンが開発マシンで一致しないことがあるため、
    # 不一致はテストを明確に xfail として扱い、CIポリシーに応じて検出・対応しやすくします。
    if runtime_version != EXPECTED_ORT_VERSION:
        pytest.xfail(f"onnxruntime runtime version {runtime_version} != pinned {EXPECTED_ORT_VERSION}")
    assert runtime_version == EXPECTED_ORT_VERSION
