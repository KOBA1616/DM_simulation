# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QGridLayout, QLabel, QCheckBox, QComboBox, QSpinBox,
    QLineEdit, QVBoxLayout, QPushButton, QFrame, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
from typing import Any
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector
from dm_toolkit.consts import ZONES, CARD_TYPES
from dm_toolkit.gui.editor.models import FilterSpec, dict_to_filterspec, filterspec_to_dict

class FilterEditorWidget(QWidget):
    """
    Reusable widget for editing FilterDef properties.
    Handles Zones, Types, Civs, Races, Costs, Powers, Flags, and Count mode.
    """

    # Signal emitted when any filter property changes
    filterChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 再発防止: min/max cost/power が入力連携(dict)のとき、
        # SpinBoxでは表現できないため内部退避して保存時に復元する。
        self._linked_range_refs: dict[str, dict[str, Any]] = {}
        self.setup_ui()

    def setup_ui(self):
        # Using a vertical layout to stack groups
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #666; font-size: 11px;")
        self.clear_btn = QPushButton(tr("Clear Filter"))
        self.clear_btn.setToolTip(tr("Reset all filter conditions"))
        safe_connect(self.clear_btn, "clicked", self.clear_filter)
        header_layout.addWidget(self.summary_label)
        header_layout.addStretch()
        header_layout.addWidget(self.clear_btn)
        main_layout.addLayout(header_layout)

        # 1. Basic Properties (Zones, Types, Civs)
        self.basic_group = QGroupBox(tr("基本フィルター"))
        basic_layout = QGridLayout(self.basic_group)
        basic_layout.setHorizontalSpacing(10)
        basic_layout.setVerticalSpacing(6)
        main_layout.addWidget(self.basic_group)

        # Zones – 1つのトグルでまとめて折り畳み
        # 再発防止: カテゴリ別2分割は UI が煩雑になるため単一 toggle にまとめた。
        #   zone_group_buttons は後方互換のため維持 (1要素のリスト)。
        self.zone_checks = {}
        # Use canonical zone list from dm_toolkit.consts to avoid duplicated definitions
        ALL_ZONES = ZONES
        zone_section = QWidget()
        zone_section_layout = QVBoxLayout(zone_section)
        zone_section_layout.setContentsMargins(0, 0, 0, 0)
        zone_section_layout.setSpacing(2)
        _zone_toggle_btn = QPushButton(tr("▶ ゾーン"))
        _zone_toggle_btn.setCheckable(True)
        _zone_toggle_btn.setChecked(False)  # 再発防止: デフォルトで閉じた状態。setVisible を明示設定すること。
        _zone_toggle_btn.setStyleSheet("text-align:left; font-weight:bold; border:none; padding:2px;")
        _zone_content = QWidget()
        _zone_content.setVisible(False)  # 初期非表示
        _zone_grid = QGridLayout(_zone_content)
        _zone_grid.setContentsMargins(12, 0, 0, 0)
        _zone_grid.setSpacing(2)
        for i, z in enumerate(ALL_ZONES):
            cb = QCheckBox(tr(z))
            cb.setToolTip(tr("ゾーン{zone}を対象選択に含める").format(zone=tr(z)))
            self.zone_checks[z] = cb
            _zone_grid.addWidget(cb, i // 2, i % 2)
            safe_connect(cb, "stateChanged", self.filterChanged.emit)
        # toggle button が押されたとき content を折り畳み / 展開
        safe_connect(_zone_toggle_btn, "toggled", _zone_content.setVisible)
        zone_section_layout.addWidget(_zone_toggle_btn)
        zone_section_layout.addWidget(_zone_content)
        self.zone_group_buttons: list = [(_zone_toggle_btn, _zone_content)]
        self.zone_label = QLabel(tr("ゾーン:"))
        # Use real Qt AlignmentFlag.AlignTop when available, otherwise fallback to 0
        align_top = getattr(Qt, 'AlignmentFlag', None)
        alignment_val = align_top.AlignTop if (align_top is not None and hasattr(align_top, 'AlignTop')) else 0
        basic_layout.addWidget(self.zone_label, 0, 0, alignment=alignment_val)
        basic_layout.addWidget(zone_section, 0, 1)

        # Types – 単一トグルで折り畳み（デフォルトで閉じた状態）
        # 再発防止: ゾーンと同様、カードタイプもトグル展開式にして UI をコンパクトに保つ。
        #   setChecked(False) で初期状態を「閉じた」にする。
        self.type_checks = {}
        _type_toggle_btn = QPushButton(tr("▶ カードタイプ"))
        _type_toggle_btn.setCheckable(True)
        _type_toggle_btn.setChecked(False)  # デフォルトで閉じた状態
        _type_toggle_btn.setStyleSheet("text-align:left; font-weight:bold; border:none; padding:2px;")
        _type_content = QWidget()
        _type_content.setVisible(False)  # 初期非表示
        _type_grid = QGridLayout(_type_content)
        _type_grid.setContentsMargins(12, 0, 0, 0)
        _type_grid.setSpacing(2)
        types = CARD_TYPES
        for i, t in enumerate(types):
            cb = QCheckBox(tr(t))
            self.type_checks[t] = cb
            _type_grid.addWidget(cb, i // 3, i % 3)
            safe_connect(cb, "stateChanged", self.filterChanged.emit)
        safe_connect(_type_toggle_btn, "toggled", _type_content.setVisible)
        _type_section = QWidget()
        _type_section_layout = QVBoxLayout(_type_section)
        _type_section_layout.setContentsMargins(0, 0, 0, 0)
        _type_section_layout.setSpacing(2)
        _type_section_layout.addWidget(_type_toggle_btn)
        _type_section_layout.addWidget(_type_content)
        self.type_label = QLabel(tr("カードタイプ:"))
        basic_layout.addWidget(self.type_label, 1, 0, alignment=alignment_val)
        basic_layout.addWidget(_type_section, 1, 1)

        # Civilizations
        self.civ_label = QLabel(tr("文明:"))
        basic_layout.addWidget(self.civ_label, 2, 0)

        civ_layout = QHBoxLayout()
        civ_layout.setContentsMargins(0, 0, 0, 0)
        self.civ_match_combo = QComboBox()
        self.civ_match_combo.addItem(tr("ANY OF (または)"), "OR")
        self.civ_match_combo.addItem(tr("ALL OF (かつ)"), "AND")
        safe_connect(self.civ_match_combo, "currentIndexChanged", self.filterChanged.emit)
        civ_layout.addWidget(self.civ_match_combo)

        self.civ_selector = CivilizationSelector(allow_multicolor=True)
        safe_connect(self.civ_selector, "changed", self.filterChanged.emit)
        civ_layout.addWidget(self.civ_selector)
        civ_layout.addStretch()

        civ_widget = QWidget()
        civ_widget.setLayout(civ_layout)
        basic_layout.addWidget(civ_widget, 2, 1)

        # Races
        self.race_label = QLabel(tr("種族:"))
        basic_layout.addWidget(self.race_label, 3, 0)

        race_layout = QHBoxLayout()
        race_layout.setContentsMargins(0, 0, 0, 0)
        self.race_match_combo = QComboBox()
        self.race_match_combo.addItem(tr("ANY OF (または)"), "OR")
        self.race_match_combo.addItem(tr("ALL OF (かつ)"), "AND")
        safe_connect(self.race_match_combo, "currentIndexChanged", self.filterChanged.emit)
        race_layout.addWidget(self.race_match_combo)

        self.races_edit = QLineEdit()
        self.races_edit.setPlaceholderText(tr("カンマ区切り (例: ドラゴン, サイバーロード)"))
        safe_connect(self.races_edit, "textChanged", self.filterChanged.emit)
        race_layout.addWidget(self.races_edit)

        race_widget = QWidget()
        race_widget.setLayout(race_layout)
        basic_layout.addWidget(race_widget, 3, 1)

        # 2. Stats (Cost, Power)
        # 再発防止: ラベルはすべて日本語で統一。英語表記を追加しないこと。
        self.stats_group = QGroupBox(tr("ステータスフィルター"))
        stats_layout = QGridLayout(self.stats_group)
        stats_layout.setHorizontalSpacing(10)
        stats_layout.setVerticalSpacing(6)
        main_layout.addWidget(self.stats_group)

        stats_layout.addWidget(QLabel(tr("コスト:")), 0, 0)
        self.min_cost_spin = QSpinBox()
        self.min_cost_spin.setRange(-1, 99) # -1 means None
        self.min_cost_spin.setValue(-1)
        self.min_cost_spin.setSpecialValueText(tr("問わない"))

        self.max_cost_spin = QSpinBox()
        self.max_cost_spin.setRange(-1, 99)
        self.max_cost_spin.setValue(-1)
        self.max_cost_spin.setSpecialValueText(tr("問わない"))

        cost_layout = QGridLayout()
        cost_layout.addWidget(QLabel(tr("最小:")), 0, 0)
        cost_layout.addWidget(self.min_cost_spin, 0, 1)
        cost_layout.addWidget(QLabel(tr("最大:")), 0, 2)
        cost_layout.addWidget(self.max_cost_spin, 0, 3)
        stats_layout.addLayout(cost_layout, 0, 1)

        # Exact cost and cost reference
        stats_layout.addWidget(QLabel(tr("コスト(完全一致):")), 1, 0)
        self.exact_cost_spin = QSpinBox()
        self.exact_cost_spin.setRange(-1, 99) # -1 means not set
        self.exact_cost_spin.setValue(-1)
        self.exact_cost_spin.setSpecialValueText(tr("未指定"))
        self.exact_cost_spin.setToolTip(tr("コストがこの値に完全一致するカードのみ対象(最小/最大より優先)"))
        stats_layout.addWidget(self.exact_cost_spin, 1, 1)

        stats_layout.addWidget(QLabel(tr("コスト参照:")), 2, 0)
        self.cost_ref_edit = QLineEdit()
        self.cost_ref_edit.setPlaceholderText(tr("変数名 (例: chosen_cost)"))
        self.cost_ref_edit.setToolTip(tr("実行コンテキストのコスト参照変数名"))
        stats_layout.addWidget(self.cost_ref_edit, 2, 1)

        safe_connect(self.exact_cost_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.cost_ref_edit, "textChanged", self.filterChanged.emit)

        safe_connect(self.min_cost_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.max_cost_spin, "valueChanged", self.filterChanged.emit)

        stats_layout.addWidget(QLabel(tr("パワー:")), 3, 0)
        self.min_power_spin = QSpinBox()
        self.min_power_spin.setRange(-1, 99999)
        self.min_power_spin.setSingleStep(500)
        self.min_power_spin.setValue(-1)
        self.min_power_spin.setSpecialValueText(tr("問わない"))

        self.max_power_spin = QSpinBox()
        self.max_power_spin.setRange(-1, 99999)
        self.max_power_spin.setSingleStep(500)
        self.max_power_spin.setValue(-1)
        self.max_power_spin.setSpecialValueText(tr("問わない"))

        power_layout = QGridLayout()
        power_layout.addWidget(QLabel(tr("最小:")), 0, 0)
        power_layout.addWidget(self.min_power_spin, 0, 1)
        power_layout.addWidget(QLabel(tr("最大:")), 0, 2)
        power_layout.addWidget(self.max_power_spin, 0, 3)
        stats_layout.addLayout(power_layout, 3, 1)

        stats_layout.addWidget(QLabel(tr("パワー(完全一致):")), 4, 0)
        self.exact_power_spin = QSpinBox()
        self.exact_power_spin.setRange(-1, 99999)
        self.exact_power_spin.setSingleStep(500)
        self.exact_power_spin.setValue(-1)
        self.exact_power_spin.setSpecialValueText(tr("未指定"))
        self.exact_power_spin.setToolTip(tr("パワーがこの値に完全一致するカードのみ対象(最小/最大より優先)"))
        stats_layout.addWidget(self.exact_power_spin, 4, 1)

        safe_connect(self.min_power_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.max_power_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.exact_power_spin, "valueChanged", self.filterChanged.emit)

        stats_layout.addWidget(QLabel(tr("数量:")), 5, 0)
        self.min_count_spin = QSpinBox()
        self.min_count_spin.setRange(-1, 99)
        self.min_count_spin.setValue(-1)
        self.min_count_spin.setSpecialValueText(tr("問わない"))

        self.max_count_spin = QSpinBox()
        self.max_count_spin.setRange(-1, 99)
        self.max_count_spin.setValue(-1)
        self.max_count_spin.setSpecialValueText(tr("問わない"))

        count_layout = QGridLayout()
        count_layout.addWidget(QLabel(tr("最小:")), 0, 0)
        count_layout.addWidget(self.min_count_spin, 0, 1)
        count_layout.addWidget(QLabel(tr("最大:")), 0, 2)
        count_layout.addWidget(self.max_count_spin, 0, 3)
        stats_layout.addLayout(count_layout, 5, 1)

        stats_layout.addWidget(QLabel(tr("数量(完全一致):")), 6, 0)
        self.exact_count_spin = QSpinBox()
        self.exact_count_spin.setRange(-1, 99)
        self.exact_count_spin.setValue(-1)
        self.exact_count_spin.setSpecialValueText(tr("未指定"))
        self.exact_count_spin.setToolTip(tr("数量がこの値に完全一致する対象のみ(最小/最大より優先)"))
        stats_layout.addWidget(self.exact_count_spin, 6, 1)

        safe_connect(self.min_count_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.max_count_spin, "valueChanged", self.filterChanged.emit)
        safe_connect(self.exact_count_spin, "valueChanged", self.filterChanged.emit)

        # 3. Flags (Tapped, Blocker, Evolution)
        # 再発防止: ラベルはすべて日本語で統一。英語表記を追加しないこと。
        self.flags_group = QGroupBox(tr("フラグフィルター"))
        flags_layout = QGridLayout(self.flags_group)
        flags_layout.setHorizontalSpacing(10)
        flags_layout.setVerticalSpacing(6)
        main_layout.addWidget(self.flags_group)

        # Helper to create tri-state combos
        def create_tristate(label):
            l = QLabel(tr(label))
            c = QComboBox()
            c.addItem(tr("問わない"), -1)
            c.addItem(tr("はい"), 1)
            c.addItem(tr("いいえ"), 0)
            safe_connect(c, "currentIndexChanged", self.filterChanged.emit)
            return l, c

        lbl_tapped, self.tapped_combo = create_tristate("タップ状態?")
        flags_layout.addWidget(lbl_tapped, 0, 0)
        flags_layout.addWidget(self.tapped_combo, 0, 1)

        lbl_blocker, self.blocker_combo = create_tristate("ブロッカー?")
        flags_layout.addWidget(lbl_blocker, 1, 0)
        flags_layout.addWidget(self.blocker_combo, 1, 1)

        lbl_evo, self.evolution_combo = create_tristate("進化クリーチャー?")
        flags_layout.addWidget(lbl_evo, 2, 0)
        flags_layout.addWidget(self.evolution_combo, 2, 1)

        lbl_card, self.card_designation_combo = create_tristate("カード指定")
        flags_layout.addWidget(lbl_card, 3, 0)
        flags_layout.addWidget(self.card_designation_combo, 3, 1)

        self.trigger_source_check = QCheckBox(tr("トリガー発生源と一致"))
        self.trigger_source_check.setToolTip(tr("イベントを発生させた特定のカード/オブジェクトを対象にします。"))
        safe_connect(self.trigger_source_check, "stateChanged", self.filterChanged.emit)
        flags_layout.addWidget(self.trigger_source_check, 4, 0, 1, 2)

        # 4. Count / Selection Mode (Keep at bottom)
        # 再発防止: ラベルはすべて日本語で統一。英語表記を追加しないこと。
        self.sel_group = QGroupBox(tr("選択設定"))
        sel_layout = QGridLayout(self.sel_group)
        sel_layout.setHorizontalSpacing(10)
        sel_layout.setVerticalSpacing(6)
        main_layout.addWidget(self.sel_group)

        self.mode_label = QLabel(tr("選択モード"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("全て"), 0)
        self.mode_combo.addItem(tr("任意"), 2)
        self.mode_combo.addItem(tr("固定数"), 1)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 99)
        self.count_spin.setToolTip(tr("選択/カウントする枚数。"))
        self.count_spin.setVisible(False) # Default hidden

        sel_layout.addWidget(self.mode_label, 0, 0)
        sel_layout.addWidget(self.mode_combo, 0, 1)
        sel_layout.addWidget(self.count_spin, 1, 1)

        safe_connect(self.mode_combo, "currentIndexChanged", self.on_mode_changed)
        safe_connect(self.count_spin, "valueChanged", self.filterChanged.emit)

        # External control label (initially hidden)
        self.external_count_label = QLabel(tr("入力変数で決定"))
        self.external_count_label.setStyleSheet("color: gray; font-style: italic;")
        self.external_count_label.setVisible(False)
        sel_layout.addWidget(self.external_count_label, 0, 1)

        # 5. Sort / Auto-Select Logic
        self.sort_mode_label = QLabel(tr("選択方法"))
        self.sort_mode_combo = QComboBox()
        self.sort_mode_combo.addItem(tr("手動（デフォルト）"), None)
        self.sort_mode_combo.addItem(tr("ランダム"), "RANDOM")
        self.sort_mode_combo.addItem(tr("最小値 (MIN)"), "MIN")
        self.sort_mode_combo.addItem(tr("最大値 (MAX)"), "MAX")
        self.sort_mode_combo.addItem(tr("全て（上書き）"), "ALL")

        self.sort_key_label = QLabel(tr("ソートキー"))
        self.sort_key_combo = QComboBox()
        self.sort_key_combo.addItem(tr("なし"), None)
        self.sort_key_combo.addItem(tr("コスト"), "COST")
        self.sort_key_combo.addItem(tr("パワー"), "POWER")

        sel_layout.addWidget(self.sort_mode_label, 2, 0)
        sel_layout.addWidget(self.sort_mode_combo, 2, 1)
        sel_layout.addWidget(self.sort_key_label, 3, 0)
        sel_layout.addWidget(self.sort_key_combo, 3, 1)

        safe_connect(self.sort_mode_combo, "currentIndexChanged", self.on_sort_mode_changed)
        safe_connect(self.sort_mode_combo, "currentIndexChanged", self.filterChanged.emit)
        safe_connect(self.sort_key_combo, "currentIndexChanged", self.filterChanged.emit)

        safe_connect(self, "filterChanged", self.update_summary_label)

        self.on_mode_changed() # Init visibility
        self.on_sort_mode_changed()
        self.update_summary_label()

    def clear_filter(self):
        """Reset all filter inputs to their defaults."""
        self.blockSignals(True)
        try:
            self._linked_range_refs.clear()
            for cb in self.zone_checks.values():
                cb.setChecked(False)
            for cb in self.type_checks.values():
                cb.setChecked(False)
            self.civ_selector.set_selected_civs([])
            self.races_edit.clear()

            self.min_cost_spin.setValue(-1)
            self.max_cost_spin.setValue(-1)
            self.exact_cost_spin.setValue(-1)
            self.cost_ref_edit.clear()
            self.min_power_spin.setValue(-1)
            self.max_power_spin.setValue(-1)
            self.exact_power_spin.setValue(-1)

            self.min_count_spin.setValue(-1)
            self.max_count_spin.setValue(-1)
            self.exact_count_spin.setValue(-1)

            self.tapped_combo.setCurrentIndex(0)
            self.blocker_combo.setCurrentIndex(0)
            self.evolution_combo.setCurrentIndex(0)
            self.card_designation_combo.setCurrentIndex(0)
            self.trigger_source_check.setChecked(False)

            self.mode_combo.setCurrentIndex(0)
            self.count_spin.setValue(1)
            self.sort_mode_combo.setCurrentIndex(0)
            self.sort_key_combo.setCurrentIndex(0)
        finally:
            self.blockSignals(False)

        # 再発防止: 一括リセット後も要約表示と外部フォーム保存を確実に同期する。
        self.on_mode_changed()
        self.on_sort_mode_changed()
        self.update_summary_label()
        self.filterChanged.emit()

    def update_summary_label(self):
        """Show a compact summary of active filter conditions."""
        filt = self.get_data()
        parts: list[str] = []

        if filt.get('zones'):
            parts.append(tr("Zones {count}").format(count=len(filt['zones'])))
        if filt.get('types'):
            parts.append(tr("Types {count}").format(count=len(filt['types'])))
        if filt.get('civilizations'):
            parts.append(tr("Civilizations {count}").format(count=len(filt['civilizations'])))
        if filt.get('races'):
            parts.append(tr("Races {count}").format(count=len(filt['races'])))

        if any(k in filt for k in ('min_cost', 'max_cost', 'exact_cost', 'cost_ref')):
            parts.append(tr("Cost"))
        if any(k in filt for k in ('min_power', 'max_power')):
            parts.append(tr("Power"))
        if any(k in filt for k in ('is_tapped', 'is_blocker', 'is_evolution', 'is_card_designation', 'is_trigger_source')):
            parts.append(tr("Flags"))
        if any(k in filt for k in ('count', 'selection_mode', 'selection_sort_key')):
            parts.append(tr("Selection"))

        if parts:
            self.summary_label.setText(
                tr("Active filters: {count}").format(count=len(parts))
                + tr(" / ")
                + tr(" ・ ").join(parts)
            )
        else:
            self.summary_label.setText(tr("No filters set"))

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_fixed = (mode == 1)
        self.count_spin.setVisible(is_fixed and self.mode_combo.isVisible())
        self.filterChanged.emit()

    def on_sort_mode_changed(self):
        mode = self.sort_mode_combo.currentData()
        # Enable key only for MIN/MAX
        needs_key = (mode == "MIN" or mode == "MAX")
        self.sort_key_combo.setEnabled(needs_key)
        self.sort_key_label.setEnabled(needs_key)

    def _set_range_field_value(self, field_name: str, spin: QSpinBox, value: Any) -> None:
        """Populate range spin while preserving input-link dict values."""
        spin.setToolTip("")
        if isinstance(value, dict):
            self._linked_range_refs[field_name] = value
            spin.setValue(-1)
            usage = value.get('input_value_usage') or value.get('input_usage') or ''
            link = value.get('input_link') or value.get('input_value_key') or ''
            spin.setToolTip(tr("入力連携: {usage} ({link})").format(usage=str(usage), link=str(link)))
            return
        self._linked_range_refs.pop(field_name, None)
        spin.setValue(value if value is not None else -1)

    def _collect_range_field(self, field_name: str, spin: QSpinBox, out: dict[str, Any]) -> None:
        """Collect range field as numeric value or preserved input-link dict."""
        val = spin.value()
        if val != -1:
            out[field_name] = val
            self._linked_range_refs.pop(field_name, None)
            return
        linked = self._linked_range_refs.get(field_name)
        if isinstance(linked, dict):
            out[field_name] = linked

    def set_external_count_control(self, active: bool):
        """
        If active, hides manual count controls and shows a label indicating
        that the count is determined by an external variable (input_value_key).
        """
        self.mode_combo.setVisible(not active)
        self.count_spin.setVisible(False) # Always hide first, then re-eval
        self.external_count_label.setVisible(active)

        if not active:
             self.on_mode_changed() # Restore state

    def set_data(self, filt_data):
        """
        Populate UI from `FilterSpec` or legacy dictionary.

        New code should prefer `set_filter_spec(FilterSpec(...))` and pass
        `FilterSpec` instances. Passing a legacy dict is still supported for
        backward-compatibility but will emit a DeprecationWarning to guide
        callers toward `FilterSpec` usage.
        """
        self.blockSignals(True)
        # Accept FilterSpec instance or legacy dict
        if isinstance(filt_data, FilterSpec):
            filt_data = filterspec_to_dict(filt_data)
        elif isinstance(filt_data, dict):
            import warnings
            warnings.warn(
                "Passing legacy dict to FilterEditorWidget.set_data is deprecated; use FilterSpec",
                DeprecationWarning,
                stacklevel=2,
            )
        if not filt_data: filt_data = {}

        # Zones
        zones = filt_data.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        # Types
        types = filt_data.get('types', [])
        for t, cb in self.type_checks.items():
            cb.setChecked(t in types)

        # Civs
        civs = filt_data.get('civilizations', [])
        self.civ_selector.set_selected_civs(civs)
        civ_match = filt_data.get('civ_match_mode', 'OR')
        idx = self.civ_match_combo.findData(civ_match)
        if idx >= 0:
            self.civ_match_combo.setCurrentIndex(idx)

        # Races
        races = filt_data.get('races', [])
        self.races_edit.setText(tr(", ").join(races))
        race_match = filt_data.get('race_match_mode', 'OR')
        idx = self.race_match_combo.findData(race_match)
        if idx >= 0:
            self.race_match_combo.setCurrentIndex(idx)

        # Costs
        self._set_range_field_value('min_cost', self.min_cost_spin, filt_data.get('min_cost'))
        self._set_range_field_value('max_cost', self.max_cost_spin, filt_data.get('max_cost'))
        self.exact_cost_spin.setValue(filt_data.get('exact_cost', -1) if filt_data.get('exact_cost') is not None else -1)
        self.cost_ref_edit.setText(filt_data.get('cost_ref', ''))

        # Powers
        self._set_range_field_value('min_power', self.min_power_spin, filt_data.get('min_power'))
        self._set_range_field_value('max_power', self.max_power_spin, filt_data.get('max_power'))
        self.exact_power_spin.setValue(filt_data.get('exact_power', -1) if filt_data.get('exact_power') is not None else -1)

        # Counts
        self.min_count_spin.setValue(filt_data.get('min_count', -1) if filt_data.get('min_count') is not None else -1)
        self.max_count_spin.setValue(filt_data.get('max_count', -1) if filt_data.get('max_count') is not None else -1)
        self.exact_count_spin.setValue(filt_data.get('exact_count', -1) if filt_data.get('exact_count') is not None else -1)

        # Flags
        def set_tristate(combo, val):
            if val is None: combo.setCurrentIndex(0) # Any
            elif val is True: combo.setCurrentIndex(1) # Yes
            else: combo.setCurrentIndex(2) # No

        set_tristate(self.tapped_combo, filt_data.get('is_tapped'))
        set_tristate(self.blocker_combo, filt_data.get('is_blocker'))
        set_tristate(self.evolution_combo, filt_data.get('is_evolution'))
        set_tristate(self.card_designation_combo, filt_data.get('is_card_designation'))
        self.trigger_source_check.setChecked(bool(filt_data.get('is_trigger_source', False)))

        # Count
        count_val = filt_data.get('count', 0)
        if count_val == 1:
            self.mode_combo.setCurrentIndex(1) # Any
            self.count_spin.setValue(1)
            self.count_spin.setVisible(False)
        elif count_val > 1:
            self.mode_combo.setCurrentIndex(2) # Fixed
            self.count_spin.setValue(count_val)
            self.count_spin.setVisible(True)
        else:
            self.mode_combo.setCurrentIndex(0) # All
            self.count_spin.setValue(1)
            self.count_spin.setVisible(False)

        # Selection Mode
        mode = filt_data.get('selection_mode')
        idx = self.sort_mode_combo.findData(mode)
        if idx < 0: idx = 0
        self.sort_mode_combo.setCurrentIndex(idx)

        # Sort Key
        key = filt_data.get('selection_sort_key')
        idx = self.sort_key_combo.findData(key)
        if idx < 0: idx = 0
        self.sort_key_combo.setCurrentIndex(idx)

        self.blockSignals(False)

    def get_data(self):
        """
        Return dictionary (FilterDef) from UI.
        Note: New API `get_filter_spec()` returns a `FilterSpec` instance.
        """
        filt: dict[str, Any] = {}

        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        if zones: filt['zones'] = zones

        types = [t for t, cb in self.type_checks.items() if cb.isChecked()]
        if types: filt['types'] = types

        civs = self.civ_selector.get_selected_civs()
        if civs:
            filt['civilizations'] = civs
            filt['civ_match_mode'] = self.civ_match_combo.currentData()

        races_str = self.races_edit.text()
        races = [r.strip() for r in races_str.split(',') if r.strip()]
        if races:
            filt['races'] = races
            filt['race_match_mode'] = self.race_match_combo.currentData()

        self._collect_range_field('min_cost', self.min_cost_spin, filt)
        self._collect_range_field('max_cost', self.max_cost_spin, filt)
        if self.exact_cost_spin.value() != -1: filt['exact_cost'] = self.exact_cost_spin.value()
        cost_ref = self.cost_ref_edit.text().strip()
        if cost_ref: filt['cost_ref'] = cost_ref

        self._collect_range_field('min_power', self.min_power_spin, filt)
        self._collect_range_field('max_power', self.max_power_spin, filt)
        if self.exact_power_spin.value() != -1: filt['exact_power'] = self.exact_power_spin.value()

        if self.min_count_spin.value() != -1: filt['min_count'] = self.min_count_spin.value()
        if self.max_count_spin.value() != -1: filt['max_count'] = self.max_count_spin.value()
        if self.exact_count_spin.value() != -1: filt['exact_count'] = self.exact_count_spin.value()

        def get_tristate(combo):
            idx = combo.currentIndex()
            if idx == 0: return None
            return (idx == 1)

        val_tapped = get_tristate(self.tapped_combo)
        if val_tapped is not None: filt['is_tapped'] = val_tapped

        val_blocker = get_tristate(self.blocker_combo)
        if val_blocker is not None: filt['is_blocker'] = val_blocker

        val_evo = get_tristate(self.evolution_combo)
        if val_evo is not None: filt['is_evolution'] = val_evo

        val_card = get_tristate(self.card_designation_combo)
        if val_card is not None: filt['is_card_designation'] = val_card

        if self.trigger_source_check.isChecked():
            filt['is_trigger_source'] = True

        mode = self.mode_combo.currentData()
        if mode == 1:
            count = self.count_spin.value()
            if count > 0: filt['count'] = count
        elif mode == 2:
            filt['count'] = 1

        sort_mode = self.sort_mode_combo.currentData()
        if sort_mode:
            filt['selection_mode'] = sort_mode
            sort_key = self.sort_key_combo.currentData()
            if sort_key and (sort_mode == "MIN" or sort_mode == "MAX"):
                filt['selection_sort_key'] = sort_key

        return filt

    def set_filter_spec(self, fs: FilterSpec):
        """Populate UI from a `FilterSpec` instance."""
        if not isinstance(fs, FilterSpec):
            raise TypeError("set_filter_spec expects a FilterSpec instance")
        self.set_data(filterspec_to_dict(fs))

    def get_filter_spec(self) -> FilterSpec:
        """Return a `FilterSpec` representing current UI state."""
        return dict_to_filterspec(self.get_data())

    def blockSignals(self, block):
        super().blockSignals(block)
        for cb in self.zone_checks.values(): cb.blockSignals(block)
        for cb in self.type_checks.values(): cb.blockSignals(block)
        self.civ_match_combo.blockSignals(block)
        self.civ_selector.blockSignals(block)
        self.race_match_combo.blockSignals(block)
        self.races_edit.blockSignals(block)
        self.min_cost_spin.blockSignals(block)
        self.max_cost_spin.blockSignals(block)
        self.exact_cost_spin.blockSignals(block)
        self.cost_ref_edit.blockSignals(block)
        self.min_power_spin.blockSignals(block)
        self.max_power_spin.blockSignals(block)
        self.exact_power_spin.blockSignals(block)
        self.min_count_spin.blockSignals(block)
        self.max_count_spin.blockSignals(block)
        self.exact_count_spin.blockSignals(block)
        self.tapped_combo.blockSignals(block)
        self.blocker_combo.blockSignals(block)
        self.evolution_combo.blockSignals(block)
        self.card_designation_combo.blockSignals(block)
        self.trigger_source_check.blockSignals(block)
        self.mode_combo.blockSignals(block)
        self.count_spin.blockSignals(block)
        self.sort_mode_combo.blockSignals(block)
        self.sort_key_combo.blockSignals(block)

    def set_visible_sections(self, sections: dict):
        """
        Toggle visibility of filter groups.
        sections: dict mapping section name ('basic', 'stats', 'flags', 'selection') to bool.
        """
        if 'basic' in sections: self.basic_group.setVisible(sections['basic'])
        if 'stats' in sections: self.stats_group.setVisible(sections['stats'])
        if 'flags' in sections: self.flags_group.setVisible(sections['flags'])
        if 'selection' in sections: self.sel_group.setVisible(sections['selection'])

    def set_allowed_fields(self, allowed_fields: list):
        """
        Toggle visibility of specific fields inside Basic group.
        allowed_fields: list of field keys ('zones', 'types', 'civilizations', 'races').
        If list is None, all are shown.
        """
        if allowed_fields is None:
            # Show all
            self.zone_label.setVisible(True)
            for cb in self.zone_checks.values(): cb.setVisible(True)
            self.type_label.setVisible(True)
            for cb in self.type_checks.values(): cb.setVisible(True)
            self.civ_label.setVisible(True)
            self.civ_selector.setVisible(True)
            self.race_label.setVisible(True)
            self.races_edit.setVisible(True)
            return

        is_allowed = lambda f: f in allowed_fields

        self.zone_label.setVisible(is_allowed('zones'))
        for cb in self.zone_checks.values():
            cb.setVisible(is_allowed('zones'))

        self.type_label.setVisible(is_allowed('types'))
        for cb in self.type_checks.values():
            cb.setVisible(is_allowed('types'))

        self.civ_label.setVisible(is_allowed('civilizations'))
        self.civ_selector.setVisible(is_allowed('civilizations'))

        self.race_label.setVisible(is_allowed('races'))
        self.races_edit.setVisible(is_allowed('races'))

    def setTitle(self, title: str):
        """
        Sets the title of the basic filter group.
        """
        self.basic_group.setTitle(title)
