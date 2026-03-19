# -*- coding: utf-8 -*-
"""契約テスト: エディタ側の統計キー一覧が共通レジストリを参照していること。"""

from dm_toolkit.gui.editor.forms.unified_widgets import make_measure_mode_combo
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_unified_widgets_uses_cardtextresources_registry():
    from dm_toolkit.gui.editor.forms.unified_widgets import MEASURE_MODE_STATS
    quick_stats = list(CardTextResources.EDITOR_QUICK_STATS_KEYS)
    assert MEASURE_MODE_STATS == quick_stats, (
        "unified_widgets の MEASURE_MODE_STATS が CardTextResources.EDITOR_QUICK_STATS_KEYS と一致しません。"
    )
