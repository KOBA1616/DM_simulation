# tests/test_headless_runner.py
"""ヘッドレスランナー結合テスト。PyQt6 不要。

再発防止: このファイルに PyQt6/PySide6 を import してはならない。
再発防止: turns はプレイヤーターン数 (active_player_id の切り替え回数) を指す。
          旧実装では resolve_command のループカウンタ(アクション数)を誤って "turns" と
          していたが、現在は "actions" キーがアクション数を保持する。
          turns の初期値は 0 のため "> 0" でなく ">= 0" でアサートする
          (ゲームが初手即終了する場合がある)。
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

# ネイティブモジュールが無い環境はスキップ
dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")
if not getattr(dm_ai_module, "IS_NATIVE", False):
    pytest.skip("Requires native dm_ai_module (IS_NATIVE=True)", allow_module_level=True)

from python.dm_env.headless_runner import run_game


def test_game_completes() -> None:
    """1ゲームが正常に完走してキーを返すことを確認する。"""
    result = run_game(deck_p0=[], deck_p1=[])
    assert "winner" in result, f"結果に 'winner' キーがない: {result}"
    assert "turns" in result, f"結果に 'turns' キーがない: {result}"
    assert "actions" in result, f"結果に 'actions' キーがない: {result}"
    # 再発防止: turns はプレイヤーターン数 (>= 0)
    assert result["turns"] >= 0, f"turns が負: {result['turns']}"
    # 再発防止: actions はアクション数 (>= turns)
    # 1ターンに複数アクションが存在するため actions >= turns が成立する
    assert result["actions"] >= result["turns"], (
        f"actions({result['actions']}) < turns({result['turns']}) は想定外"
    )


def test_json_output(tmp_path: Path) -> None:
    """output_json を指定した場合に有効な JSON が書き出されることを確認する。"""
    out = str(tmp_path / "result.json")
    run_game([], [], output_json=out)
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert "winner" in data, f"JSON に 'winner' キーがない: {data}"


def test_batch_10_games() -> None:
    """10ゲームを連続実行してすべて結果を返すことを確認する。"""
    results = [run_game([], []) for _ in range(10)]
    assert all("winner" in r for r in results), (
        "一部のゲームで 'winner' キーが返されなかった"
    )


def test_seeded_reproducibility() -> None:
    """同一シードで2回実行して同じ勝者になることを確認する。"""
    r1 = run_game([], [], seed=42)
    r2 = run_game([], [], seed=42)
    assert r1["winner"] == r2["winner"], (
        f"シードが同じなのに結果が異なる: {r1['winner']} vs {r2['winner']}"
    )
