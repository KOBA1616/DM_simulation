# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QComboBox, QSpinBox, QPushButton, QLabel, QHBoxLayout
from dm_toolkit.gui.localization import tr
from dm_toolkit.consts import ZONES_EXTENDED


def make_scope_combo(parent=None):
    combo = QComboBox(parent)
    scopes = ["PLAYER_SELF","PLAYER_OPPONENT","TARGET_SELECT","ALL_PLAYERS","RANDOM","ALL_FILTERED","NONE"]
    for s in scopes:
        combo.addItem(tr(s), s)
    return combo


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
