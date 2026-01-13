# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QDialog,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QInputDialog, QListWidget, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer
import json
import os
import random
import dm_ai_module
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.widgets.card_action_dialog import CardActionDialog

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

        # 0.5 Board Control
        board_group = QGroupBox(tr("Board Control"))
        board_layout = QVBoxLayout()

        self.btn_clear_board = QPushButton(tr("Clear Board"))
        self.btn_clear_board.clicked.connect(self.on_clear_board)
        board_layout.addWidget(self.btn_clear_board)

        self.btn_reset_game = QPushButton(tr("Reset Game"))
        self.btn_reset_game.clicked.connect(self.on_reset_game)
        board_layout.addWidget(self.btn_reset_game)

        board_group.setLayout(board_layout)
        layout.addWidget(board_group)

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

        # 2.5 Game Flow Control
        flow_group = QGroupBox(tr("Game Flow"))
        flow_layout = QVBoxLayout()

        self.btn_end_turn = QPushButton(tr("End Turn"))
        self.btn_end_turn.clicked.connect(self.on_end_turn)
        flow_layout.addWidget(self.btn_end_turn)

        self.btn_draw_card = QPushButton(tr("Draw Card"))
        self.btn_draw_card.clicked.connect(self.on_draw_card)
        flow_layout.addWidget(self.btn_draw_card)

        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)

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
        self.parent_window.reset_game()

        # 2. Clear all zones completely (safe method: move to deck bottom)
        self._clear_board_safely()

        # 3. Apply scenario config
        config = scenario.get("config", {})
        
        p0 = self.gs.players[0]
        p1 = self.gs.players[1]

        def setup_zone(pid, zone_enum, card_ids):
            """Setup a zone with specified cards, moving from deck if needed."""
            if not card_ids: 
                return

            # Find max instance ID
            max_id = self._get_max_instance_id()

            for i, cid in enumerate(card_ids):
                iid = max_id + 1 + i
                # Add card to specified zone
                self._add_card_to_zone(pid, zone_enum, cid, iid)
                # Remove from deck if it exists there
                self._remove_from_deck(pid, cid)

        # Apply Config
        if "my_hand_cards" in config: 
            setup_zone(0, dm_ai_module.Zone.HAND, config["my_hand_cards"])
        if "my_battle_zone" in config: 
            setup_zone(0, dm_ai_module.Zone.BATTLE, config["my_battle_zone"])
        if "my_mana_zone" in config: 
            setup_zone(0, dm_ai_module.Zone.MANA, config["my_mana_zone"])
        if "my_shields" in config: 
            setup_zone(0, dm_ai_module.Zone.SHIELD, config["my_shields"])
        if "my_grave_yard" in config: 
            setup_zone(0, dm_ai_module.Zone.GRAVEYARD, config["my_grave_yard"])

        if "enemy_hand_cards" in config: 
            setup_zone(1, dm_ai_module.Zone.HAND, config["enemy_hand_cards"])
        if "enemy_battle_zone" in config: 
            setup_zone(1, dm_ai_module.Zone.BATTLE, config["enemy_battle_zone"])
        if "enemy_mana_zone" in config: 
            setup_zone(1, dm_ai_module.Zone.MANA, config["enemy_mana_zone"])
        if "enemy_shields" in config: 
            setup_zone(1, dm_ai_module.Zone.SHIELD, config["enemy_shields"])

        self.parent_window.update_ui()
        loaded_name = scenario.get('name') or "?"
        QMessageBox.information(
            self,
            tr("Scenario Loaded"),
            tr("Loaded scenario: {name}").format(name=loaded_name),
        )

    def _clear_board_safely(self):
        """Clear all cards from Hand, Battle, Mana, Shield zones (move to deck bottom)."""
        if not self.gs:
            return
        
        for pid in [0, 1]:
            p = self.gs.players[pid]
            # Collect all cards from target zones
            cards_to_clear = []
            for card in list(p.hand):
                cards_to_clear.append((card, dm_ai_module.Zone.HAND))
            for card in list(p.battle_zone):
                cards_to_clear.append((card, dm_ai_module.Zone.BATTLE))
            for card in list(p.mana_zone):
                cards_to_clear.append((card, dm_ai_module.Zone.MANA))
            for card in list(p.shield_zone):
                cards_to_clear.append((card, dm_ai_module.Zone.SHIELD))
            
            # Move each to graveyard (safe, no side effects)
            for card, from_zone in cards_to_clear:
                try:
                    dm_ai_module.DevTools.move_cards(
                        self.gs, card.instance_id, from_zone, dm_ai_module.Zone.GRAVEYARD
                    )
                except Exception:
                    pass  # Ignore move failures

    def _get_max_instance_id(self):
        """Get the maximum instance ID across all cards."""
        max_id = 0
        for p in [self.gs.players[0], self.gs.players[1]]:
            for zone in [p.hand, p.mana_zone, p.battle_zone, p.shield_zone, p.graveyard, p.deck]:
                for c in zone:
                    if c.instance_id > max_id:
                        max_id = c.instance_id
        return max_id

    def _add_card_to_zone(self, pid, zone_enum, cid, iid):
        """Add a card to a specific zone."""
        if zone_enum == dm_ai_module.Zone.HAND:
            self.gs.add_card_to_hand(pid, cid, iid)
        elif zone_enum == dm_ai_module.Zone.MANA:
            self.gs.add_card_to_mana(pid, cid, iid)
        elif zone_enum == dm_ai_module.Zone.BATTLE:
            self.gs.add_test_card_to_battle(pid, cid, iid, False, True)
        elif zone_enum == dm_ai_module.Zone.SHIELD:
            self.gs.add_card_to_hand(pid, cid, iid)
            dm_ai_module.DevTools.move_cards(
                self.gs, iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.SHIELD
            )
        elif zone_enum == dm_ai_module.Zone.GRAVEYARD:
            self.gs.add_card_to_hand(pid, cid, iid)
            dm_ai_module.DevTools.move_cards(
                self.gs, iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.GRAVEYARD
            )
        elif zone_enum == dm_ai_module.Zone.DECK:
            self.gs.add_card_to_deck(pid, cid, iid)

    def _remove_from_deck(self, pid, card_id):
        """Remove a specific card from player deck if present."""
        if pid >= len(self.gs.players):
            return
        p = self.gs.players[pid]
        p.deck = [c for c in p.deck if c.card_id != card_id]

    def on_add_specific_card(self):
        """Open card action dialog for adding cards."""
        if not self.gs:
            return
        
        dialog = CardActionDialog(
            parent=self.parent_window,
            game_state=self.gs,
            card_db=self.card_db,
            action_type="ADD_CARD"
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_result()
            if result:
                self._execute_add_card_action(result)

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
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("Failed to add card: {error}").format(error=e),
            )

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

    def on_clear_board(self):
        """Clear all cards from Hand, Battle, Mana, Shield zones."""
        if not self.gs:
            return
        
        reply = QMessageBox.question(
            self,
            tr("Clear Board"),
            tr("Clear all cards from Hand, Battle, Mana, and Shield zones?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._clear_board_safely()
            if self.parent_window:
                self.parent_window.update_ui()
            QMessageBox.information(self, tr("Info"), tr("Board cleared."))

    def on_reset_game(self):
        """Reset entire game state."""
        if not self.parent_window:
            return
        
        reply = QMessageBox.question(
            self,
            tr("Reset Game"),
            tr("Reset the game to initial state?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.parent_window.reset_game()

    def on_end_turn(self):
        """End the current turn (advance phase)."""
        if not self.gs:
            return
        
        try:
            # Call the phase manager to advance
            dm_ai_module.PhaseManager.advance_phase(self.gs)
            if self.parent_window:
                self.parent_window.update_ui()
            QMessageBox.information(self, tr("Info"), tr("Turn advanced."))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to advance turn: {e}")

    def on_draw_card(self):
        """Draw a card for the selected player."""
        if not self.gs:
            return
        
        pid = self.target_player_combo.currentData()
        player = self.gs.players[pid]
        
        if not player.deck:
            QMessageBox.warning(self, tr("Info"), tr("Deck is empty."))
            return
        
        try:
            # Draw one card
            card = player.deck[-1]
            player.deck.pop()
            player.hand.append(card)
            
            if self.parent_window:
                self.parent_window.update_ui()
            QMessageBox.information(self, tr("Info"), tr("Card drawn."))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to draw card: {e}")

    def _execute_add_card_action(self, result):
        """Execute card addition action from dialog result."""
        if not result:
            return
        
        player_id = result["player_id"]
        zone = result["zone"]
        card_ids = result["card_ids"]
        quantity = result["quantity"]
        
        if not self.gs or player_id >= len(self.gs.players):
            return
        
        try:
            max_id = self._get_max_instance_id()
            
            for card_id in card_ids:
                for i in range(quantity):
                    iid = max_id + 1 + i
                    self._add_card_to_zone(player_id, zone, card_id, iid)
                    self._remove_from_deck(player_id, card_id)
                    max_id = iid
            
            if self.parent_window:
                self.parent_window.update_ui()
            
            QMessageBox.information(
                self,
                tr("Success"),
                tr("Added {count} card(s) to {zone}").format(
                    count=len(card_ids) * quantity,
                    zone=self.zone_combo.currentText() if hasattr(self, 'zone_combo') else "zone"
                )
            )
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Failed to add cards: {e}")

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
            QMessageBox.information(
                self,
                tr("Success"),
                tr("Scenario '{name}' saved.").format(name=name),
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("Failed to save: {error}").format(error=e),
            )

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
