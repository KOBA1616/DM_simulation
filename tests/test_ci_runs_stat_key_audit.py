# -*- coding: utf-8 -*-
"""Contract test: CI workflow must run the stat key audit script.

RED step: ensure repository CI invokes `tools/stat_key_audit.py` as part of
the stat contract tests workflow.
"""
from pathlib import Path


def test_github_actions_runs_stat_key_audit():
    p = Path('.github/workflows/stat-contract-tests.yml')
    assert p.exists(), 'CI workflow .github/workflows/stat-contract-tests.yml が見つかりません'
    txt = p.read_text(encoding='utf-8')
    assert 'tools/stat_key_audit.py' in txt, 'CI ワークフローが tools/stat_key_audit.py を実行していません'
