import re
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_mutate_tap_all_and_specific():
    out_all = CardTextGenerator._mutate_tap("対象", 0, "体", "", "", False)
    assert "すべてタップ" in out_all or "すべてタップする" in out_all

    out_some = CardTextGenerator._mutate_tap("対象", 2, "体", "", "", False)
    assert re.search(r"2.?体|2.?体選び", out_some)
    assert "タップ" in out_some


def test_mutate_untap_all_and_specific():
    out_all = CardTextGenerator._mutate_untap("対象", 0, "体", "", "", False)
    assert "アンタップ" in out_all

    out_some = CardTextGenerator._mutate_untap("対象", 3, "体", "", "", False)
    assert re.search(r"3.?体|3.?体選び", out_some)
    assert "アンタップ" in out_some


def test_mutate_power_positive_and_negative():
    out_pos = CardTextGenerator._mutate_power("対象", 5, "", "", "", False)
    assert "パワー" in out_pos and "+5" in out_pos or "5" in out_pos

    out_neg = CardTextGenerator._mutate_power("対象", -2, "", "", "", False)
    assert "パワー" in out_neg and "-2" in out_neg


def test_mutate_add_keyword_shows_grant():
    # Provide a known keyword id; output should indicate granting a keyword
    out = CardTextGenerator._mutate_add_keyword("対象", 1, "体", "", "speed_attacker", True)
    assert "与える" in out or "与える。" in out or "を与える" in out


def test_mutate_remove_keyword():
    out = CardTextGenerator._mutate_remove_keyword("対象", 0, "", "", "shield_burn", False)
    assert "無視" in out or "無視する" in out or "取り除" in out


def test_mutate_add_passive_and_modifier():
    out1 = CardTextGenerator._mutate_add_passive("対象", 0, "", "", "just_diver", False)
    assert ("与える" in out1) or ("パッシブ" in out1)

    # ADD_MODIFIER uses same handler as ADD_PASSIVE_EFFECT
    out2 = CardTextGenerator._mutate_add_passive("対象", 0, "", "", "", False)
    assert ("与える" in out2) or ("パッシブ" in out2)


def test_mutate_add_cost_modifier():
    out = CardTextGenerator._mutate_add_cost("対象", 1, "体", "", "", False)
    assert "コスト" in out or "コスト修正" in out


def test_mutate_give_power_and_give_ability_aliases():
    # GIVE_POWER is mapped to _mutate_power
    out = CardTextGenerator._mutate_power("対象", 4, "", "", "", False)
    assert "パワー" in out and ("4" in out or "+4" in out)

    # GIVE_ABILITY uses same behavior as add_keyword mapping
    out2 = CardTextGenerator._mutate_add_keyword("対象", 1, "体", "", "blocker", True)
    assert "与える" in out2 or "を与える" in out2
