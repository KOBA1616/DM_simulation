# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QComboBox, QSpinBox, QPushButton, QLabel, QHBoxLayout, QWidget, QGridLayout, QCheckBox
from dm_toolkit.gui.i18n import tr
from dm_toolkit.consts import ZONES_EXTENDED, TargetScope


def make_scope_combo(parent=None, include_zones=False):
    """
    スコープ選択のコンボボックスを作成。
    
    Args:
        include_zones: Trueの場合、BATTLE_ZONE等のゾーン指定も選択肢に含める
    
    Note: Uses TargetScope.SELF/OPPONENT for player scopes (unified constants).
    """
    combo = QComboBox(parent)
    # Use unified TargetScope constants
    scopes = [
        TargetScope.SELF,  # "SELF" instead of "PLAYER_SELF"
        TargetScope.OPPONENT,  # "OPPONENT" instead of "PLAYER_OPPONENT"
        "TARGET_SELECT",
        "ALL_PLAYERS",
        "RANDOM",
        "ALL_FILTERED",
        "NONE"
    ]
    
    if include_zones:
        # ゾーン指定を追加（CIP効果と同様の選択肢）
        zone_scopes = ["BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"]
        scopes = scopes + zone_scopes
    
    for s in scopes:
        combo.addItem(tr(s), s)
    return combo


def make_player_scope_selector(parent=None):
    """文明選択のような形式で、自分/相手のみ選択できるスコープUI。

    - UIはチェックボックス2つ（排他）
    - 値は TargetScope に対応（SELF / OPPONENT）
    
    Note: Now uses TargetScope.SELF/OPPONENT (unified constants).
    """
    w = QWidget(parent)
    layout = QGridLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(12)

    # Use TargetScope constants for display
    self_cb = QCheckBox(tr(TargetScope.SELF), w)
    opp_cb = QCheckBox(tr(TargetScope.OPPONENT), w)

    layout.addWidget(self_cb, 0, 0)
    layout.addWidget(opp_cb, 0, 1)

    return w, self_cb, opp_cb


def make_value_spin(parent=None, minimum=-9999, maximum=9999):
    spin = QSpinBox(parent)
    spin.setRange(minimum, maximum)
    return spin


def make_measure_mode_combo(parent=None):
    combo = QComboBox(parent)
    combo.addItem(tr("CARDS_MATCHING_FILTER"), "CARDS_MATCHING_FILTER")
    stats = ["MANA_CIVILIZATION_COUNT", "SHIELD_COUNT", "HAND_COUNT", "CARDS_DRAWN_THIS_TURN"]
    for s in stats:
        combo.addItem(tr(s), s)
    return combo


def make_ref_mode_combo(parent=None):
    combo = QComboBox(parent)
    ref_modes = ["SYM_CREATURE", "SYM_SPELL", "G_ZERO", "HYPER_ENERGY", "NONE"]
    for r in ref_modes:
        combo.addItem(tr(r), r)
    return combo


def make_zone_combos(parent=None):
    # Uses ZONES_EXTENDED which now favors COMMAND_ZONES (BATTLE, MANA, etc)
    # but also includes extended options like DECK_BOTTOM
    src = QComboBox(parent)
    dst = QComboBox(parent)
    for z in ZONES_EXTENDED:
        src.addItem(tr(z), z)
        dst.addItem(tr(z), z)
    return src, dst


def make_option_controls(parent=None):
    count_spin = QSpinBox(parent)
    count_spin.setRange(1, 10)
    count_spin.setValue(1)
    gen_btn = QPushButton(tr("Generate Options"), parent)
    label = QLabel(tr("Options to Add"), parent)
    layout = QHBoxLayout()
    layout.addWidget(count_spin)
    layout.addWidget(gen_btn)
    return count_spin, gen_btn, label, layout
