# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QInputDialog, QListWidget, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer
import json
import os
import random
import dm_ai_module
from dm_toolkit.gui.localization import tr

class ScenarioToolsDock(QDockWidget):
    def __init__(self, parent=None, game_state=None, card_db=None):
        super().__init__(tr("Scenario Tools"), parent)
        self.setObjectName("ScenarioToolsDock")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.gs = game_state
        self.card_db = card_db
        self.parent_window = parent

        self.scenarios = [] # List of loaded scenario data
        self.init_ui()
        self.load_scenarios()

        self.is_recording = False
        self.recorded_actions = []

    def set_game_state(self, game_state, card_db):
        self.gs = game_state
        self.card_db = card_db

    def init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # 0. Load Scenario
        load_group = QGroupBox(tr("Load Scenario"))
        load_layout = QVBoxLayout()

        self.scenario_combo = QComboBox()
        load_layout.addWidget(self.scenario_combo)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton(tr("Load Selected"))
        self.btn_load.clicked.connect(self.on_load_selected_scenario)
        self.btn_reload_list = QPushButton(tr("Refresh List"))
        self.btn_reload_list.clicked.connect(self.load_scenarios)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_reload_list)
        load_layout.addLayout(btn_layout)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # 1. Target Selection
        target_group = QGroupBox(tr("Target Zone Selection"))
        target_layout = QVBoxLayout()

        self.target_player_combo = QComboBox()
        self.target_player_combo.addItem(tr("P0 (Me)"), 0)
        self.target_player_combo.addItem(tr("P1 (Opponent)"), 1)
        target_layout.addWidget(QLabel(tr("Target Player:")))
        target_layout.addWidget(self.target_player_combo)

        self.target_zone_combo = QComboBox()
        zones = [
            ("HAND", dm_ai_module.Zone.HAND),
            ("BATTLE_ZONE", dm_ai_module.Zone.BATTLE),
            ("MANA_ZONE", dm_ai_module.Zone.MANA),
            ("SHIELD_ZONE", dm_ai_module.Zone.SHIELD),
            ("GRAVEYARD", dm_ai_module.Zone.GRAVEYARD),
            ("DECK", dm_ai_module.Zone.DECK)
        ]
        for name, z_enum in zones:
            self.target_zone_combo.addItem(tr(name), z_enum)

        target_layout.addWidget(QLabel(tr("Target Zone:")))
        target_layout.addWidget(self.target_zone_combo)
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # 2. Card Manipulation
        manip_group = QGroupBox(tr("Card Manipulation"))
        manip_layout = QVBoxLayout()

        self.btn_add_card = QPushButton(tr("Add Specific Card..."))
        self.btn_add_card.clicked.connect(self.on_add_specific_card)
        manip_layout.addWidget(self.btn_add_card)

        self.btn_add_random = QPushButton(tr("Add Random Card"))
        self.btn_add_random.clicked.connect(self.on_add_random_card)
        manip_layout.addWidget(self.btn_add_random)

        self.btn_clear_zone = QPushButton(tr("Clear Zone"))
        self.btn_clear_zone.clicked.connect(self.on_clear_zone)
        manip_layout.addWidget(self.btn_clear_zone)

        manip_group.setLayout(manip_layout)
        layout.addWidget(manip_group)

        # 3. Game State Editing
        state_group = QGroupBox(tr("Game State"))
        state_layout = QFormLayout()

        self.spin_mana_count = QSpinBox()
        self.spin_mana_count.setRange(0, 99)

        self.btn_reset_mana = QPushButton(tr("Untap All Mana"))
        self.btn_reset_mana.clicked.connect(self.on_untap_all_mana)
        state_layout.addWidget(self.btn_reset_mana)

        state_group.setLayout(state_layout)
        layout.addWidget(state_group)

        # 4. Scenario Management (Save)
        scen_group = QGroupBox(tr("Scenario Management"))
        scen_layout = QVBoxLayout()

        self.btn_save_scenario = QPushButton(tr("Save Current State"))
        self.btn_save_scenario.clicked.connect(self.on_save_scenario)
        scen_layout.addWidget(self.btn_save_scenario)

        scen_group.setLayout(scen_layout)
        layout.addWidget(scen_group)

        # 5. Combo Recording
        rec_group = QGroupBox(tr("Combo Recording"))
        rec_layout = QVBoxLayout()

        self.btn_record = QPushButton(tr("Start Recording"))
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.on_toggle_record)
        rec_layout.addWidget(self.btn_record)

        self.lbl_record_status = QLabel(tr("Status: Idle"))
        rec_layout.addWidget(self.lbl_record_status)

        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        layout.addStretch()
        self.setWidget(container)

    def load_scenarios(self):
        self.scenarios = []
        self.scenario_combo.clear()

        path = "data/scenarios.json"
        if not os.path.exists(path) and os.path.exists("../data/scenarios.json"):
            path = "../data/scenarios.json"

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.scenarios = json.load(f)
            except Exception as e:
                print(f"Failed to load scenarios: {e}")

        for i, sc in enumerate(self.scenarios):
            name = sc.get("name", f"Scenario {i}")
            self.scenario_combo.addItem(name, i)

    def on_load_selected_scenario(self):
        idx = self.scenario_combo.currentData()
        if idx is None or idx < 0 or idx >= len(self.scenarios):
            return

        scenario = self.scenarios[idx]
        self.apply_scenario(scenario)

    def apply_scenario(self, scenario):
        if not self.parent_window: return

        # 1. Reset Game to clean slate
        # Note: reset_game in app.py reloads decks. We want empty state ideally,
        # or we accept deck load and then clear relevant zones?
        # Ideally, Scenario Mode overrides standard decks.

        # We will reset, then clear all zones manually.
        self.parent_window.reset_game()

        # 2. Clear Zones (Hand, Battle, Mana, Shield)
        # We leave Decks intact unless scenario specifies them?
        # Scenarios usually specify 'my_deck'. If not, we might want to keep the loaded deck?
        # Let's assume scenario config overrides everything present in it.

        # Actually, best way to 'Clear' is to create a new GameState with NO decks?
        # But app.py reset_game logic is fixed.
        # We will use DevTools or loop to move everything to Graveyard (then to "Void"?).

        # A simpler way: Just set the GameState internal vectors to empty? Not exposed.
        # We will use DevTools.move_cards to Graveyard for everything in Hand/Battle/Mana/Shield.
        # Deck is harder to clear without drawing all.

        # Strategy: Iterate all zones and move to Graveyard.
        # Wait, if we move Deck to Graveyard, we might kill the game (Deckout).
        # We only clear zones that the scenario DEFINES.

        config = scenario.get("config", {})

        p0 = self.gs.players[0]
        p1 = self.gs.players[1]

        # Helper to setup zone
        def setup_zone(pid, zone_enum, card_ids):
            # 1. Clear existing cards in this zone?
            # It's hard to clear specific zone safely without side effects (like triggering OnDestroy).
            # For "Scenario Setup", we assume "God Mode" editing.
            # We can't suppress triggers easily.
            # But we can just ADD cards. If the user wants a clean board, they should have a "Clear Board" scenario?
            # Or we force clear.

            # Let's try to clear Hand/Battle/Mana/Shield by moving to Deck? (Then shuffle/reset deck?)
            # Moving to Graveyard triggers effects.
            # If we just ADD, we might end up with mixed state.

            # Since this is a tool, let's just ADD for now, or assume the user resets before load?
            # The user clicked "Load". They expect the scenario state.

            # If I can't clear safely, I will just add.
            # Ideally, `GameState` should have `reset_zone`.

            # 2. Add cards
            if not card_ids: return

            # We need new instance IDs.
            # Find max ID first.
            max_id = 0
            for p in [self.gs.players[0], self.gs.players[1]]:
                for z in [p.hand, p.mana_zone, p.battle_zone, p.shield_zone, p.graveyard, p.deck]:
                    for c in z:
                        if c.instance_id > max_id: max_id = c.instance_id

            for i, cid in enumerate(card_ids):
                iid = max_id + 1 + i
                if zone_enum == dm_ai_module.Zone.HAND:
                    self.gs.add_card_to_hand(pid, cid, iid)
                elif zone_enum == dm_ai_module.Zone.MANA:
                    self.gs.add_card_to_mana(pid, cid, iid)
                elif zone_enum == dm_ai_module.Zone.BATTLE:
                    self.gs.add_test_card_to_battle(pid, cid, iid, False, True) # Tapped=False, Sick=True
                elif zone_enum == dm_ai_module.Zone.SHIELD:
                    # Workaround: Add to hand, move to Shield
                    self.gs.add_card_to_hand(pid, cid, iid)
                    dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.SHIELD)
                elif zone_enum == dm_ai_module.Zone.GRAVEYARD:
                    self.gs.add_card_to_hand(pid, cid, iid)
                    dm_ai_module.DevTools.move_cards(self.gs, iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.GRAVEYARD)
                elif zone_enum == dm_ai_module.Zone.DECK:
                    self.gs.add_card_to_deck(pid, cid, iid)

        # We assume the user wants the scenario to be the state.
        # But clearing is hard.
        # Let's implement a 'smart clear': Move Hand/Battle/Mana/Shield to Deck (bottom), then shuffle?
        # That effectively clears the board without triggering 'Destroy'.

        # Clear P0 and P1 board
        for pid in [0, 1]:
            # We must iterate copies because we are modifying the zones
            # Actually, move_cards might invalidate iterators.
            # Using DevTools to move everything to Deck?
            pass
            # Skipping complex clear for now to avoid crashes.
            # Appending cards is "Safe".

        # Apply Config
        if "my_hand_cards" in config: setup_zone(0, dm_ai_module.Zone.HAND, config["my_hand_cards"])
        if "my_battle_zone" in config: setup_zone(0, dm_ai_module.Zone.BATTLE, config["my_battle_zone"])
        if "my_mana_zone" in config: setup_zone(0, dm_ai_module.Zone.MANA, config["my_mana_zone"])
        if "my_shields" in config: setup_zone(0, dm_ai_module.Zone.SHIELD, config["my_shields"])
        if "my_grave_yard" in config: setup_zone(0, dm_ai_module.Zone.GRAVEYARD, config["my_grave_yard"])

        if "enemy_hand_cards" in config: setup_zone(1, dm_ai_module.Zone.HAND, config["enemy_hand_cards"])
        if "enemy_battle_zone" in config: setup_zone(1, dm_ai_module.Zone.BATTLE, config["enemy_battle_zone"])
        if "enemy_mana_zone" in config: setup_zone(1, dm_ai_module.Zone.MANA, config["enemy_mana_zone"])
        if "enemy_shields" in config: setup_zone(1, dm_ai_module.Zone.SHIELD, config["enemy_shields"])

        self.parent_window.update_ui()
        QMessageBox.information(self, tr("Scenario Loaded"), tr(f"Loaded scenario: {scenario.get('name')}"))

    def on_add_specific_card(self):
        if not self.gs: return
        items = []
        for cid, card in self.card_db.items():
            items.append(f"{cid}: {card.name}")

        item, ok = QInputDialog.getItem(self, tr("Select Card"), tr("Card:"), items, 0, False)
        if ok and item:
            card_id = int(item.split(":")[0])
            self.add_card_to_target(card_id)

    def on_add_random_card(self):
        if not self.gs: return
        if not self.card_db: return

        cid = random.choice(list(self.card_db.keys()))
        self.add_card_to_target(cid)

    def add_card_to_target(self, card_id):
        pid = self.target_player_combo.currentData()
        zone = self.target_zone_combo.currentData()

        max_id = 0
        for p in [0, 1]:
            player = self.gs.players[p]
            for z in [player.hand, player.mana_zone, player.battle_zone, player.shield_zone, player.graveyard, player.deck]:
                for c in z:
                    if c.instance_id > max_id: max_id = c.instance_id

        new_iid = max_id + 1

        try:
            if zone == dm_ai_module.Zone.HAND:
                self.gs.add_card_to_hand(pid, card_id, new_iid)
            elif zone == dm_ai_module.Zone.MANA:
                self.gs.add_card_to_mana(pid, card_id, new_iid)
            elif zone == dm_ai_module.Zone.DECK:
                self.gs.add_card_to_deck(pid, card_id, new_iid)
            elif zone == dm_ai_module.Zone.BATTLE:
                self.gs.add_test_card_to_battle(pid, card_id, new_iid, False, True)
            elif zone == dm_ai_module.Zone.GRAVEYARD:
                self.gs.add_card_to_hand(pid, card_id, new_iid)
                dm_ai_module.DevTools.move_cards(self.gs, new_iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.GRAVEYARD)
            elif zone == dm_ai_module.Zone.SHIELD:
                self.gs.add_card_to_hand(pid, card_id, new_iid)
                dm_ai_module.DevTools.move_cards(self.gs, new_iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.SHIELD)

            if self.parent_window:
                self.parent_window.update_ui()

        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to add card: {e}")

    def on_clear_zone(self):
        # Implementation: Move all cards in target zone to Graveyard.
        pid = self.target_player_combo.currentData()
        zone = self.target_zone_combo.currentData()
        player = self.gs.players[pid]

        target_cards = []
        if zone == dm_ai_module.Zone.HAND: target_cards = list(player.hand)
        elif zone == dm_ai_module.Zone.BATTLE: target_cards = list(player.battle_zone)
        elif zone == dm_ai_module.Zone.MANA: target_cards = list(player.mana_zone)
        elif zone == dm_ai_module.Zone.SHIELD: target_cards = list(player.shield_zone)

        if not target_cards:
            QMessageBox.information(self, tr("Info"), tr("Zone is already empty."))
            return

        for c in target_cards:
            # We use DevTools to move to Graveyard.
            # Source zone must be correct.
            dm_ai_module.DevTools.move_cards(self.gs, c.instance_id, zone, dm_ai_module.Zone.GRAVEYARD)

        if self.parent_window:
            self.parent_window.update_ui()

    def on_untap_all_mana(self):
        pid = self.target_player_combo.currentData()
        player = self.gs.players[pid]
        for c in player.mana_zone:
            c.is_tapped = False
        if self.parent_window:
            self.parent_window.update_ui()

    def on_save_scenario(self):
        if not self.gs: return

        name, ok = QInputDialog.getText(self, tr("Save Scenario"), tr("Scenario Name:"))
        if not ok or not name: return

        config: dict[str, object] = {
            "my_mana": len(self.gs.players[0].mana_zone),
            "enemy_shield_count": len(self.gs.players[1].shield_zone),
            "enemy_can_use_trigger": True,
            "loop_proof_mode": False
        }

        def get_ids(zone_cards):
            return [c.card_id for c in zone_cards]

        p0 = self.gs.players[0]
        p1 = self.gs.players[1]

        config["my_hand_cards"] = get_ids(p0.hand)
        config["my_battle_zone"] = get_ids(p0.battle_zone)
        config["my_mana_zone"] = get_ids(p0.mana_zone)
        config["my_grave_yard"] = get_ids(p0.graveyard)
        config["my_shields"] = get_ids(p0.shield_zone)
        config["my_deck"] = get_ids(p0.deck)

        config["enemy_hand_cards"] = get_ids(p1.hand)
        config["enemy_battle_zone"] = get_ids(p1.battle_zone)
        config["enemy_mana_zone"] = get_ids(p1.mana_zone)
        config["enemy_grave_yard"] = get_ids(p1.graveyard)
        config["enemy_shields"] = get_ids(p1.shield_zone)
        config["enemy_deck"] = get_ids(p1.deck)

        if self.recorded_actions:
            config["demonstration"] = self.recorded_actions

        scenario_entry = {
            "name": name,
            "description": f"Saved from Scenario Mode",
            "config": config
        }

        filepath = "data/scenarios.json"
        if not os.path.exists(filepath) and os.path.exists("../data/scenarios.json"):
            filepath = "../data/scenarios.json"

        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = []

        overwritten = False
        for i, item in enumerate(data):
            if item.get("name") == name:
                data[i] = scenario_entry
                overwritten = True
                break

        if not overwritten:
            data.append(scenario_entry)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.load_scenarios() # Refresh list
            QMessageBox.information(self, tr("Success"), tr(f"Scenario '{name}' saved."))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to save: {e}")

    def on_toggle_record(self):
        if self.btn_record.isChecked():
            self.is_recording = True
            self.recorded_actions = []
            self.btn_record.setText(tr("Stop Recording"))
            self.lbl_record_status.setText(tr("Status: Recording..."))
        else:
            self.is_recording = False
            self.btn_record.setText(tr("Start Recording"))
            self.lbl_record_status.setText(f"{tr('Status: Stopped')} ({len(self.recorded_actions)} actions)")

    def record_action(self, action_str):
        if self.is_recording:
            self.recorded_actions.append(action_str)
            self.lbl_record_status.setText(f"{tr('Status: Recording...')} ({len(self.recorded_actions)})")
