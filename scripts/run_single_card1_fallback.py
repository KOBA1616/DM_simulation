import os
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ['DM_DISABLE_NATIVE'] = '1'

import pytest

if __name__ == '__main__':
    rc = pytest.main(['-q','tests/test_card1_hand_quality.py::TestCard1HandQuality::test_select_target_appears_after_draw'])
    raise SystemExit(rc)
