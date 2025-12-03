import json
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QPushButton, QMessageBox, QFormLayout, QWidget, QCheckBox, QGridLayout, QScrollArea, QTabWidget, QGroupBox, QListWidget, QListWidgetItem,
    QStackedWidget, QTextEdit
)
from gui.widgets.card_widget import CardWidget

# Localization Dictionary
JP_TEXT = {
    "window_title": "デュエル・マスターズ カードエディタ (JSON)",
    "basic_info": "基本情報",
    "effects": "効果 (詳細)",
    "preview": "プレビュー",
    "save_all": "保存",
    "close": "閉じる",
    "new_card": "新規カード",
    "delete_card": "カード削除",
    "id": "ID",
    "name": "名前",
    "civilization": "文明",
    "type": "タイプ",
    "cost": "コスト",
    "power": "パワー",
    "races": "種族 (カンマ区切り)",
    "keywords": "キーワード能力 (PASSIVE_CONSTとして保存)",
    "trigger": "トリガー条件",
    "action": "処理 (アクション)",
    "target": "対象",
    "filter_zone": "対象ゾーン",
    "filter_civ": "対象文明",
    "filter_race": "対象種族",
    "filter_type": "対象タイプ",
    "count": "対象数 / 数値",
    "add_effect": "効果を追加",
    "remove_effect": "効果を削除",
    "update_effect": "効果を更新",
    "effect_list": "効果リスト",
    "action_type": "処理タイプ",
    "scope": "対象範囲",
    "target_player": "対象プレイヤー",
    # Enums / Values
    "ON_PLAY": "出た時 (ON_PLAY)",
    "ON_ATTACK": "攻撃する時 (ON_ATTACK)",
    "ON_DESTROY": "破壊された時 (ON_DESTROY)",
    "PASSIVE_CONST": "常在効果 (PASSIVE_CONST)",
    "SHIELD_TRIGGER": "S・トリガー (keyword)",
    "DESTROY": "破壊する (DESTROY)",
    "RETURN_TO_HAND": "手札に戻す (RETURN_TO_HAND)",
    "ADD_MANA": "マナ加速 (ADD_MANA)",
    "DRAW_CARD": "ドロー (DRAW_CARD)",
    "SEARCH_DECK_BOTTOM": "山札を見て手札/下へ (SEARCH_DECK_BOTTOM)",
    "MEKRAID": "メクレイド (MEKRAID)",
    "TAP": "タップする (TAP)",
    "UNTAP": "アンタップする (UNTAP)",
    "COST_REFERENCE": "コスト参照/軽減 (COST_REFERENCE)",
    "BATTLE_ZONE": "バトルゾーン",
    "MANA_ZONE": "マナゾーン",
    "HAND": "手札",
    "GRAVEYARD": "墓地",
    "DECK": "山札",
    "SHIELD_ZONE": "シールドゾーン",
    "CREATURE": "クリーチャー",
    "SPELL": "呪文",
    "EVOLUTION_CREATURE": "進化クリーチャー",
    "LIGHT": "光",
    "WATER": "水",
    "DARKNESS": "闇",
    "FIRE": "火",
    "NATURE": "自然",
    "ZERO": "ゼロ",
    "PLAYER_SELF": "自分",
    "PLAYER_OPPONENT": "相手",
    "TARGET_SELECT": "選択して対象",
    "ALL": "全て",
    # Keywords
    "BLOCKER": "ブロッカー",
    "SPEED_ATTACKER": "スピードアタッカー",
    "SLAYER": "スレイヤー",
    "DOUBLE_BREAKER": "W・ブレイカー",
    "TRIPLE_BREAKER": "T・ブレイカー",
    "POWER_ATTACKER": "パワーアタッカー",
    "EVOLUTION": "進化",
    "MACH_FIGHTER": "マッハファイター",
    "G_STRIKE": "G・ストライク"
}

def tr(key):
    return JP_TEXT.get(key, key)

class CardEditor(QDialog):
    def __init__(self, json_path, parent=None):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle(tr("window_title"))
        self.resize(1000, 700)
        self.cards_data = []
        self.current_card_index = -1
        self.load_data()
        self.init_ui()

    def load_data(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.cards_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")
                self.cards_data = []
        else:
            self.cards_data = []

    def save_data(self):
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.cards_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", "Cards saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON: {e}")

    def init_ui(self):
        # Left: Card List
        list_layout = QVBoxLayout()
        self.card_list = QListWidget()
        self.card_list.currentRowChanged.connect(self.load_selected_card)
        list_layout.addWidget(self.card_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton(tr("new_card"))
        add_btn.clicked.connect(self.create_new_card)
        del_btn = QPushButton(tr("delete_card"))
        del_btn.clicked.connect(self.delete_current_card)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        list_layout.addLayout(btn_layout)

        # Middle: Form (Tabs)
        self.tabs = QTabWidget()

        # Tab 1: Basic Info & Keywords
        self.basic_tab = QWidget()
        self.setup_basic_tab()
        self.tabs.addTab(self.basic_tab, tr("basic_info"))

        # Tab 2: Effects (Visual Builder)
        self.effects_tab = QWidget()
        self.setup_effects_tab()
        self.tabs.addTab(self.effects_tab, tr("effects"))

        # Right: Preview
        preview_layout = QVBoxLayout()
        preview_label = QLabel(tr("preview"))
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_container = QWidget()
        self.preview_container_layout = QVBoxLayout(self.preview_container)
        self.preview_card = None

        preview_layout.addWidget(self.preview_container)
        preview_layout.addStretch()

        # Bottom Buttons
        action_layout = QHBoxLayout()
        save_btn = QPushButton(tr("save_all"))
        save_btn.clicked.connect(self.save_data)
        close_btn = QPushButton(tr("close"))
        close_btn.clicked.connect(self.reject)
        action_layout.addWidget(save_btn)
        action_layout.addWidget(close_btn)

        # Assemble Layouts
        top_layout = QHBoxLayout()
        top_layout.addLayout(list_layout, 1)
        top_layout.addWidget(self.tabs, 3)
        top_layout.addLayout(preview_layout, 1)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(top_layout)
        root_layout.addLayout(action_layout)

        self.refresh_list()

    def setup_basic_tab(self):
        layout = QVBoxLayout(self.basic_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.id_input = QSpinBox()
        self.id_input.setRange(1, 9999)
        form.addRow(tr("id") + ":", self.id_input)

        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self.update_preview)
        self.name_input.textChanged.connect(self.update_current_card_data)
        form.addRow(tr("name") + ":", self.name_input)

        self.civ_input = QComboBox()
        self.civ_input.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        self.civ_input.currentTextChanged.connect(self.update_preview)
        self.civ_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow(tr("civilization") + ":", self.civ_input)

        self.type_input = QComboBox()
        self.type_input.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
        self.type_input.currentTextChanged.connect(self.update_current_card_data)
        form.addRow(tr("type") + ":", self.type_input)

        self.cost_input = QSpinBox()
        self.cost_input.setRange(0, 99)
        self.cost_input.valueChanged.connect(self.update_preview)
        self.cost_input.valueChanged.connect(self.update_current_card_data)
        form.addRow(tr("cost") + ":", self.cost_input)

        self.power_input = QSpinBox()
        self.power_input.setRange(0, 99999)
        self.power_input.setSingleStep(500)
        self.power_input.valueChanged.connect(self.update_preview)
        self.power_input.valueChanged.connect(self.update_current_card_data)
        form.addRow(tr("power") + ":", self.power_input)

        self.races_input = QLineEdit()
        self.races_input.setPlaceholderText(tr("races"))
        self.races_input.textChanged.connect(self.update_current_card_data)
        form.addRow(tr("races") + ":", self.races_input)

        # Keywords
        keywords_label = QLabel(tr("keywords") + ":")
        form.addRow(keywords_label)
        
        self.keywords_layout = QGridLayout()
        self.keyword_checkboxes = {}
        keywords_list = [
            "BLOCKER", "SPEED_ATTACKER", "SLAYER",
            "DOUBLE_BREAKER", "TRIPLE_BREAKER", "POWER_ATTACKER",
            "EVOLUTION", "MACH_FIGHTER", "G_STRIKE"
        ]
        
        for i, kw in enumerate(keywords_list):
            cb = QCheckBox(tr(kw))
            cb.stateChanged.connect(self.update_current_card_data)
            self.keyword_checkboxes[kw] = cb
            self.keywords_layout.addWidget(cb, i // 2, i % 2)
            
        form.addRow(self.keywords_layout)

        # Shield Trigger (Special Keyword)
        self.shield_trigger_cb = QCheckBox(tr("SHIELD_TRIGGER"))
        self.shield_trigger_cb.stateChanged.connect(self.update_current_card_data)
        form.addRow(self.shield_trigger_cb)

        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

    def setup_effects_tab(self):
        layout = QVBoxLayout(self.effects_tab)

        # Split: List of Effects (Top) and Effect Detail Editor (Bottom)

        # Top: List
        list_group = QGroupBox(tr("effect_list"))
        list_layout = QVBoxLayout(list_group)
        self.effects_list = QListWidget()
        self.effects_list.currentRowChanged.connect(self.load_selected_effect)
        list_layout.addWidget(self.effects_list)

        btn_layout = QHBoxLayout()
        add_eff_btn = QPushButton(tr("add_effect"))
        add_eff_btn.clicked.connect(self.create_new_effect)
        rem_eff_btn = QPushButton(tr("remove_effect"))
        rem_eff_btn.clicked.connect(self.remove_effect)
        btn_layout.addWidget(add_eff_btn)
        btn_layout.addWidget(rem_eff_btn)
        list_layout.addLayout(btn_layout)

        layout.addWidget(list_group, 1)

        # Bottom: Editor
        editor_group = QGroupBox("効果詳細設定 (Visual Editor)")
        editor_layout = QVBoxLayout(editor_group)

        # Trigger
        trig_layout = QHBoxLayout()
        trig_layout.addWidget(QLabel(tr("trigger") + ":"))
        self.eff_trigger_combo = QComboBox()
        triggers = ["ON_PLAY", "ON_ATTACK", "ON_DESTROY", "PASSIVE_CONST"]
        for t in triggers:
            self.eff_trigger_combo.addItem(tr(t), t)
        trig_layout.addWidget(self.eff_trigger_combo)
        editor_layout.addLayout(trig_layout)

        # Action List inside Effect
        self.eff_action_list_widget = QListWidget()
        self.eff_action_list_widget.setFixedHeight(80)
        self.eff_action_list_widget.currentRowChanged.connect(self.load_selected_action)
        editor_layout.addWidget(QLabel("この効果のアクション一覧:"))
        editor_layout.addWidget(self.eff_action_list_widget)

        act_btn_layout = QHBoxLayout()
        add_act_btn = QPushButton("アクション追加")
        add_act_btn.clicked.connect(self.add_action_to_effect)
        rem_act_btn = QPushButton("アクション削除")
        rem_act_btn.clicked.connect(self.remove_action_from_effect)
        act_btn_layout.addWidget(add_act_btn)
        act_btn_layout.addWidget(rem_act_btn)
        editor_layout.addLayout(act_btn_layout)

        # Action Detail Editor
        self.action_detail_group = QGroupBox("アクション設定")
        detail_form = QFormLayout(self.action_detail_group)

        self.act_type_combo = QComboBox()
        actions = ["DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DRAW_CARD", "SEARCH_DECK_BOTTOM", "MEKRAID", "TAP", "UNTAP", "COST_REFERENCE", "NONE"]
        for a in actions:
            self.act_type_combo.addItem(tr(a), a)
        detail_form.addRow(tr("action_type") + ":", self.act_type_combo)

        self.act_scope_combo = QComboBox()
        scopes = ["PLAYER_SELF", "PLAYER_OPPONENT", "TARGET_SELECT", "ALL", "NONE"]
        for s in scopes:
            self.act_scope_combo.addItem(tr(s), s)
        detail_form.addRow(tr("scope") + ":", self.act_scope_combo)

        # Filter Settings
        filter_box = QGroupBox("フィルター / 対象条件")
        filter_layout = QGridLayout(filter_box)

        filter_layout.addWidget(QLabel(tr("filter_zone") + ":"), 0, 0)
        self.zone_checks = {}
        zones = ["BATTLE_ZONE", "MANA_ZONE", "HAND", "GRAVEYARD", "SHIELD_ZONE", "DECK"]
        zones_layout = QGridLayout()
        for i, z in enumerate(zones):
            cb = QCheckBox(tr(z))
            self.zone_checks[z] = cb
            zones_layout.addWidget(cb, i // 3, i % 3)
        filter_layout.addLayout(zones_layout, 0, 1)

        filter_layout.addWidget(QLabel(tr("target_player") + ":"), 1, 0)
        self.filter_player_combo = QComboBox()
        self.filter_player_combo.addItems(["NONE", "SELF", "OPPONENT"])
        filter_layout.addWidget(self.filter_player_combo, 1, 1)

        filter_layout.addWidget(QLabel(tr("filter_type") + ":"), 2, 0)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["NONE", "CREATURE", "SPELL"])
        filter_layout.addWidget(self.filter_type_combo, 2, 1)

        filter_layout.addWidget(QLabel(tr("count") + ":"), 3, 0)
        self.filter_count_spin = QSpinBox()
        self.filter_count_spin.setRange(0, 20)
        filter_layout.addWidget(self.filter_count_spin, 3, 1)
        
        detail_form.addRow(filter_box)

        # Generic Values
        self.val1_spin = QSpinBox()
        self.val1_spin.setRange(0, 99)
        detail_form.addRow("Value 1 (枚数/コスト等):", self.val1_spin)

        self.str_val_edit = QLineEdit()
        detail_form.addRow("String Value (Keyword等):", self.str_val_edit)

        # Apply Button
        apply_btn = QPushButton("現在のアクション設定を適用")
        apply_btn.clicked.connect(self.apply_action_changes)
        detail_form.addRow(apply_btn)

        editor_layout.addWidget(self.action_detail_group)
        layout.addWidget(editor_group, 2)

    def refresh_list(self):
        self.card_list.clear()
        for card in self.cards_data:
            item = QListWidgetItem(f"{card.get('id')} - {card.get('name')}")
            item.setData(Qt.ItemDataRole.UserRole, card.get('id'))
            self.card_list.addItem(item)

    def create_new_card(self):
        new_id = 1
        if self.cards_data:
            new_id = max(c.get('id', 0) for c in self.cards_data) + 1
        
        new_card = {
            "id": new_id,
            "name": "New Card",
            "civilization": "FIRE",
            "type": "CREATURE",
            "cost": 1,
            "power": 1000,
            "races": [],
            "effects": []
        }
        self.cards_data.append(new_card)
        self.refresh_list()
        self.card_list.setCurrentRow(self.card_list.count() - 1)

    def delete_current_card(self):
        row = self.card_list.currentRow()
        if row >= 0:
            del self.cards_data[row]
            self.refresh_list()
            self.current_card_index = -1
            # Clear UI

    def load_selected_card(self, row):
        if row < 0 or row >= len(self.cards_data):
            return
        
        self.current_card_index = row
        card = self.cards_data[row]
        
        self.block_signals_recursive(True)

        self.id_input.setValue(card.get('id', 0))
        self.name_input.setText(card.get('name', ''))
        self.civ_input.setCurrentText(card.get('civilization', 'FIRE'))
        self.type_input.setCurrentText(card.get('type', 'CREATURE'))
        self.cost_input.setValue(card.get('cost', 0))
        self.power_input.setValue(card.get('power', 0))
        
        races = card.get('races', [])
        self.races_input.setText(", ".join(races))
        
        # Keywords check (PASSIVE_CONST sync)
        effects = card.get('effects', [])
        active_keywords = set()
        for eff in effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                for act in eff.get('actions', []):
                    if 'str_val' in act:
                        active_keywords.add(act['str_val'])
        
        for kw, cb in self.keyword_checkboxes.items():
            cb.setChecked(kw in active_keywords)

        # Shield Trigger check (Root keyword)
        st_kw = card.get('keywords', {}).get('shield_trigger', False)
        self.shield_trigger_cb.setChecked(st_kw)

        # Load Effects List
        self.refresh_effects_list()

        self.block_signals_recursive(False)
        self.update_preview()

    def block_signals_recursive(self, block):
        self.id_input.blockSignals(block)
        self.name_input.blockSignals(block)
        self.civ_input.blockSignals(block)
        self.type_input.blockSignals(block)
        self.cost_input.blockSignals(block)
        self.power_input.blockSignals(block)
        self.races_input.blockSignals(block)
        for cb in self.keyword_checkboxes.values():
            cb.blockSignals(block)
        self.shield_trigger_cb.blockSignals(block)

    def update_current_card_data(self):
        if self.current_card_index < 0:
            return

        card = self.cards_data[self.current_card_index]

        card['id'] = self.id_input.value()
        card['name'] = self.name_input.text()
        card['civilization'] = self.civ_input.currentText()
        card['type'] = self.type_input.currentText()
        card['cost'] = self.cost_input.value()
        card['power'] = self.power_input.value()

        races_str = self.races_input.text()
        card['races'] = [r.strip() for r in races_str.split(',')] if races_str.strip() else []

        # Sync Shield Trigger
        if self.shield_trigger_cb.isChecked():
            if 'keywords' not in card: card['keywords'] = {}
            card['keywords']['shield_trigger'] = True
        else:
            if 'keywords' in card and 'shield_trigger' in card['keywords']:
                del card['keywords']['shield_trigger']

        # Sync Keyword Checkboxes to PASSIVE_CONST
        new_effects = []
        existing_effects = card.get('effects', [])

        known_keywords = set(self.keyword_checkboxes.keys())

        # Keep existing non-keyword effects
        for eff in existing_effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                actions_to_keep = []
                for act in eff.get('actions', []):
                    if act.get('str_val') not in known_keywords:
                        actions_to_keep.append(act)
                if actions_to_keep:
                    eff['actions'] = actions_to_keep
                    new_effects.append(eff)
            else:
                new_effects.append(eff)

        # Add checked keywords
        active_kws = []
        for kw, cb in self.keyword_checkboxes.items():
            if cb.isChecked():
                active_kws.append(kw)

        if active_kws:
            actions = []
            for kw in active_kws:
                actions.append({
                    "type": "NONE",
                    "scope": "NONE",
                    "filter": {},
                    "value1": 0,
                    "value2": 0,
                    "str_val": kw
                })
            new_effects.append({
                "trigger": "PASSIVE_CONST",
                "condition": {"type": "NONE", "value": 0, "str_val": ""},
                "actions": actions
            })

        card['effects'] = new_effects

        # UI Refresh
        item = self.card_list.item(self.current_card_index)
        if item:
            item.setText(f"{card['id']} - {card['name']}")

        # Note: We do NOT call refresh_effects_list() here to avoid resetting the effect editor
        # when editing basic info. Effect list only needs refresh if keywords changed.
        # But we did just rebuild `effects`, so if keywords were added/removed, the list *is* stale regarding PASSIVE_CONST.
        # Ideally, we check if the PASSIVE_CONST structure actually changed.
        # For MVP, we will only refresh if we are NOT on the effects tab? Or simply let it be.
        # If user is in Effects tab, adding a keyword in Basic tab will update the internal data,
        # but the ListWidget might show old count.
        # Since Basic and Effects are separate tabs, user won't see the glitch unless they switch tabs.
        # When switching tabs, we could refresh.
        # For now, let's call it ONLY if we detected a keyword change? Hard to track.
        # Let's call it, but realize it might lose selection.
        # Since Basic Info is on a different tab than Effects, losing selection in Effects tab is acceptable.
        if self.tabs.currentWidget() == self.effects_tab:
             self.refresh_effects_list()

    def update_preview(self):
        for i in reversed(range(self.preview_container_layout.count())):
            item = self.preview_container_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        name = self.name_input.text() or "New Card"
        cost = self.cost_input.value()
        power = self.power_input.value()
        civ = self.civ_input.currentText()
        
        self.preview_card = CardWidget(0, name, cost, power, civ)
        self.preview_container_layout.addWidget(self.preview_card)

    # --- Effects Logic ---

    def refresh_effects_list(self):
        if self.current_card_index < 0: return
        self.effects_list.clear()
        effects = self.cards_data[self.current_card_index].get('effects', [])
        for i, eff in enumerate(effects):
            trig = eff.get('trigger', 'NONE')
            action_count = len(eff.get('actions', []))
            self.effects_list.addItem(f"#{i}: {tr(trig)} ({action_count} actions)")

    def create_new_effect(self):
        if self.current_card_index < 0: return
        new_eff = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "actions": []
        }
        self.cards_data[self.current_card_index]['effects'].append(new_eff)
        self.refresh_effects_list()
        self.effects_list.setCurrentRow(self.effects_list.count() - 1)

    def remove_effect(self):
        if self.current_card_index < 0: return
        row = self.effects_list.currentRow()
        if row >= 0:
            del self.cards_data[self.current_card_index]['effects'][row]
            self.refresh_effects_list()

    def load_selected_effect(self, row):
        if row < 0:
            self.eff_trigger_combo.setEnabled(False)
            self.eff_action_list_widget.clear()
            return

        self.eff_trigger_combo.setEnabled(True)
        effect = self.cards_data[self.current_card_index]['effects'][row]

        # Set Trigger
        # Use findData if we had set item data, but we didn't for triggers in previous code.
        # We need to set up trigger combo to have user data too.
        # In setup_effects_tab we did: self.eff_trigger_combo.addItem(tr(t), t)

        trig_val = effect.get('trigger', 'ON_PLAY')
        idx = self.eff_trigger_combo.findData(trig_val)
        if idx >= 0: self.eff_trigger_combo.setCurrentIndex(idx)
        else: self.eff_trigger_combo.setCurrentIndex(0)

        # Connect trigger change
        self.eff_trigger_combo.currentIndexChanged.disconnect() if self.eff_trigger_combo.receivers(self.eff_trigger_combo.currentIndexChanged) else None
        self.eff_trigger_combo.currentIndexChanged.connect(lambda: self.update_effect_trigger(row))

        # Load Actions
        self.eff_action_list_widget.clear()
        for i, act in enumerate(effect.get('actions', [])):
            act_type = act.get('type', 'NONE')
            self.eff_action_list_widget.addItem(f"{i}: {tr(act_type)}")

    def update_effect_trigger(self, row):
        # use currentData()
        new_trig = self.eff_trigger_combo.currentData()
        self.cards_data[self.current_card_index]['effects'][row]['trigger'] = new_trig
        # Refresh label in list
        item = self.effects_list.item(row)
        if item:
            action_count = len(self.cards_data[self.current_card_index]['effects'][row]['actions'])
            item.setText(f"#{row}: {tr(new_trig)} ({action_count} actions)")

    def add_action_to_effect(self):
        row = self.effects_list.currentRow()
        if row < 0: return

        new_action = {
            "type": "DESTROY",
            "scope": "TARGET_SELECT",
            "value1": 1,
            "filter": {"zones": ["BATTLE_ZONE"], "count": 1}
        }
        self.cards_data[self.current_card_index]['effects'][row]['actions'].append(new_action)
        self.load_selected_effect(row) # Refresh action list
        self.eff_action_list_widget.setCurrentRow(self.eff_action_list_widget.count() - 1)

    def remove_action_from_effect(self):
        eff_row = self.effects_list.currentRow()
        act_row = self.eff_action_list_widget.currentRow()
        if eff_row >= 0 and act_row >= 0:
            del self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row]
            self.load_selected_effect(eff_row)

    def load_selected_action(self, act_row):
        eff_row = self.effects_list.currentRow()
        if eff_row < 0 or act_row < 0: return

        action = self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row]

        # Populate form
        idx = self.act_type_combo.findData(action.get('type', 'DESTROY'))
        if idx >= 0: self.act_type_combo.setCurrentIndex(idx)

        idx = self.act_scope_combo.findData(action.get('scope', 'TARGET_SELECT'))
        if idx >= 0: self.act_scope_combo.setCurrentIndex(idx)

        # Filter
        filt = action.get('filter', {})
        zones = filt.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        tp = filt.get('owner', 'NONE') # Use owner as per previous fix
        # But wait, previous data used target_player in some cases?
        # Let's check both for safety when loading, but save as owner.
        if tp == 'NONE' or tp is None:
             tp = filt.get('target_player', 'NONE')

        self.filter_player_combo.setCurrentText(tp if isinstance(tp, str) else 'NONE')

        types = filt.get('types', [])
        ftype = 'NONE'
        if 'CREATURE' in types: ftype = 'CREATURE'
        if 'SPELL' in types: ftype = 'SPELL'
        self.filter_type_combo.setCurrentText(ftype)

        self.filter_count_spin.setValue(filt.get('count', 1))

        self.val1_spin.setValue(action.get('value1', 0))
        self.str_val_edit.setText(action.get('str_val', ''))

    def apply_action_changes(self):
        eff_row = self.effects_list.currentRow()
        act_row = self.eff_action_list_widget.currentRow()
        if eff_row < 0 or act_row < 0: return

        # Build action object
        new_act = {
            "type": self.act_type_combo.currentData(),
            "scope": self.act_scope_combo.currentData(),
            "value1": self.val1_spin.value(),
            "str_val": self.str_val_edit.text()
        }

        # Build filter
        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        target_player = self.filter_player_combo.currentText()
        card_type = self.filter_type_combo.currentText()

        filt = {}
        if zones: filt['zones'] = zones
        if target_player != "NONE": filt['owner'] = target_player
        if card_type != "NONE": filt['types'] = [card_type]
        count = self.filter_count_spin.value()
        if count > 0: filt['count'] = count

        new_act['filter'] = filt

        self.cards_data[self.current_card_index]['effects'][eff_row]['actions'][act_row] = new_act

        # Refresh list label
        self.eff_action_list_widget.currentItem().setText(f"{act_row}: {tr(new_act['type'])}")
        QMessageBox.information(self, "Success", "Action updated!")

