# -*- coding: utf-8 -*-
import sys
import os
import random
import json
import csv
from typing import Any, List, Optional, Dict, cast
from types import ModuleType

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox, QSplitter,
    QCheckBox, QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QDockWidget, QTabWidget,
    QInputDialog, QToolBar
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer

dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.types import GameState, CardDB, Action, PlayerID, CardID
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command
from dm_toolkit.gui.deck_builder import DeckBuilder
from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.simulation_dialog import SimulationDialog
from dm_toolkit.gui.widgets.log_viewer import LogViewer
from dm_toolkit.gui.game_session import GameSession
from dm_toolkit.gui.input_handler import GameInputHandler
from dm_toolkit.gui.layout_builder import LayoutBuilder

class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("DM AI Simulator"))
        self.resize(1600, 900)

        # Initialize GameSession (Logic)
        self.session = GameSession(
            callback_update_ui=self.update_ui,
            callback_log=self.log_message,
            callback_input_request=self.handle_user_input_request,
            callback_action_executed=self.on_action_executed
        )

        # Initialize Input Handler
        self.input_handler = GameInputHandler(self, self.session)

        # Ensure log list exists before any callbacks
        self.log_viewer = LogViewer()

        # Load card database
        self.card_db: CardDB = {}
        loaded_db = EngineCompat.JsonLoader_load_cards("data/cards.json")
        if loaded_db:
             self.card_db = loaded_db
        else:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                json_path = os.path.join(base_dir, 'data', 'cards.json')
                with open(json_path, 'r', encoding='utf-8') as _f:
                    self.card_db = json.load(_f)
            except Exception:
                self.card_db = {}
        
        # Initialize Game
        self.session.initialize_game(self.card_db)

        self.p0_deck_ids: Optional[List[int]] = None
        self.p1_deck_ids: Optional[List[int]] = None
        self.last_command_index: int = 0

        # Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.session.step_phase)
        # Note: Session manages 'is_processing', App manages 'is_running' for timer
        self.is_running: bool = False

        # Initialize Layout using Builder
        self.layout_builder = LayoutBuilder(self)
        self.layout_builder.build()

        self.update_ui()
        self.showMaximized()

    @property
    def gs(self):
        return self.session.gs

    def log_message(self, msg: str) -> None:
        self.log_viewer.log_message(msg)

    def on_action_executed(self, action_dict: Dict[str, Any]) -> None:
        """Callback from GameSession when an action is executed."""
        if hasattr(self, 'loop_recorder'):
            self.loop_recorder.record_action(str(action_dict))
        if hasattr(self, 'scenario_tools'):
            self.scenario_tools.record_action(str(action_dict))

    def open_deck_builder(self) -> None:
        self.deck_builder = DeckBuilder(self.card_db)
        self.deck_builder.show()

    def open_card_editor(self) -> None:
        self.card_editor = CardEditor("data/cards.json")
        self.card_editor.data_saved.connect(self.reload_card_data)
        self.card_editor.show()

    def reload_card_data(self) -> None:
        try:
            loaded = EngineCompat.JsonLoader_load_cards("data/cards.json")
            if loaded is not None:
                self.card_db = loaded
            if hasattr(self, 'deck_builder') and self.deck_builder.isVisible():
                self.deck_builder.reload_database()
            if hasattr(self, 'scenario_tools'):
                self.scenario_tools.set_game_state(self.gs, self.card_db)
            self.log_viewer.log_message(tr("Card Data Reloaded from Editor Save"))
        except Exception as e:
            self.log_viewer.log_message(f"{tr('Error reloading cards')}: {e}")

    def toggle_scenario_mode(self, checked: bool) -> None:
        if not hasattr(self, 'scenario_tools'): return
        if checked:
            self.scenario_tools.show()
            self.log_viewer.log_message(tr("Scenario Mode Enabled"))
            if self.is_running:
                self.toggle_simulation()
        else:
            self.scenario_tools.hide()
            self.log_viewer.log_message(tr("Scenario Mode Disabled"))

    def open_simulation_dialog(self) -> None:
        self.sim_dialog = SimulationDialog(self.card_db, self)
        self.sim_dialog.show()

    def load_deck_p0(self) -> None:
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(self, tr("Load Deck P0"), "data/decks", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f: deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                self.p0_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P0')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def load_deck_p1(self) -> None:
        os.makedirs("data/decks", exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(self, tr("Load Deck P1"), "data/decks", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f: deck_ids = json.load(f)
                if len(deck_ids) != 40:
                    QMessageBox.warning(self, tr("Invalid Deck"), tr("Deck must have 40 cards."))
                    return
                self.p1_deck_ids = deck_ids
                self.reset_game()
                self.log_viewer.log_message(f"{tr('Loaded Deck for P1')}: {os.path.basename(fname)}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load deck')}: {e}")

    def show_help(self) -> None:
        QMessageBox.information(self, tr("Help / Manual"), tr("Help text..."))

    def toggle_simulation(self) -> None:
        if self.is_running:
            self.timer.stop()
            if hasattr(self, 'control_panel'):
                self.control_panel.set_start_button_text(tr("Start Sim"))
            self.is_running = False
        else:
            self.timer.start(500)
            if hasattr(self, 'control_panel'):
                self.control_panel.set_start_button_text(tr("Stop Sim"))
            self.is_running = True

    def reset_game(self) -> None:
        self.timer.stop()
        self.is_running = False
        self.input_handler.selected_targets = []
        if hasattr(self, 'control_panel'):
            self.control_panel.set_confirm_button_visible(False)
            self.control_panel.set_pass_button_visible(False)
            self.control_panel.set_start_button_text(tr("Start Sim"))
        self.log_viewer.clear_logs()

        self.session.reset_game(self.p0_deck_ids, self.p1_deck_ids)

        if hasattr(self, 'scenario_tools'):
            self.scenario_tools.set_game_state(self.gs, self.card_db)
        self.last_command_index = 0
        self.update_ui()

    def pass_turn(self) -> None:
        if hasattr(self, 'current_pass_action') and self.current_pass_action:
            self.session.execute_action(self.current_pass_action)

    def confirm_selection(self) -> None:
        self.input_handler.confirm_selection()

    def on_card_clicked(self, card_id: int, instance_id: int) -> None:
        self.input_handler.on_card_clicked(card_id, instance_id)

    def on_card_double_clicked(self, card_id: int, instance_id: int) -> None:
        self.input_handler.on_card_double_clicked(card_id, instance_id)

    def on_card_hovered(self, card_id: int) -> None:
        if card_id >= 0:
            card_data = self._get_card_by_id(card_id)
            if card_data and hasattr(self, 'card_detail_panel'):
                self.card_detail_panel.update_card(card_data)

    def _get_card_by_id(self, card_id: int):
        db = self.card_db
        if db is None: return None
        try:
            if isinstance(db, dict): return db.get(card_id)
        except Exception: pass
        try:
            if hasattr(db, '__contains__') and card_id in db: return db[card_id]
        except Exception: pass
        for meth in ('get_card', 'get_by_id'):
            try:
                fn = getattr(db, meth, None)
                if callable(fn): return fn(card_id)
            except Exception: pass
        try:
            for k in getattr(db, 'keys', lambda: [])():
                if int(k) == int(card_id): return db[k]
        except Exception: pass
        return None

    def on_resolve_effect_from_stack(self, index: int) -> None:
        self.input_handler.on_resolve_effect_from_stack(index)

    def handle_user_input_request(self) -> None:
        """Called by GameSession when input is needed."""
        self.input_handler.handle_user_input_request()

    def update_ui(self) -> None:
        if self.gs is None:
            return
        if not hasattr(self, 'turn_label'):
            return

        turn_number = EngineCompat.get_turn_number(self.gs)
        current_phase = EngineCompat.get_current_phase(self.gs)

        phase_map = {
            "START": "Start Phase",
            "DRAW": "Draw Phase",
            "MANA": "Mana Phase",
            "MAIN": "Main Phase",
            "ATTACK": "Attack Phase",
            "BLOCK": "Block Phase",
            "END": "End Phase"
        }
        phase_key = phase_map.get(str(current_phase), str(current_phase))

        active_pid = EngineCompat.get_active_player_id(self.gs)
        self.turn_label.setText(tr("Turn: {turn}").format(turn=turn_number))
        self.phase_label.setText(tr("Phase: {phase}").format(phase=tr(phase_key)))
        self.active_label.setText(tr("Active: P{player_id}").format(player_id=active_pid))
        
        self.stack_view.update_state(self.gs, self.card_db)
        self.effect_debugger.update_state(self.gs, self.card_db)

        history = EngineCompat.get_command_history(self.gs)
        try: current_len = len(history)
        except: current_len = 0
        if current_len > self.last_command_index:
            for i in range(self.last_command_index, current_len):
                cmd = history[i]
                try: desc = describe_command(cmd, self.gs, self.card_db)
                except: desc = str(cmd)
                self.log_viewer.log_message(desc)
            self.last_command_index = current_len

        class _P:
             hand: List[Any] = []
             mana_zone: List[Any] = []
             battle_zone: List[Any] = []
             shield_zone: List[Any] = []
             graveyard: List[Any] = []
             deck: List[Any] = []

        p0 = EngineCompat.get_player(self.gs, 0) or _P()
        p1 = EngineCompat.get_player(self.gs, 1) or _P()
        
        legal_actions = []
        if active_pid == 0 and self.control_panel.is_p0_human() and not self.gs.game_over:
             from dm_toolkit.commands import generate_legal_commands
             legal_actions = generate_legal_commands(self.gs, self.card_db)

        self.current_pass_action = None
        for cmd in legal_actions:
            try: d = cmd.to_dict()
            except: d = {}
            if d.get('type') == 'PASS' or d.get('legacy_original_type') == 'PASS':
                self.current_pass_action = cmd
                break

        self.control_panel.set_pass_button_visible(bool(self.current_pass_action))

        god_view = self.control_panel.is_god_view()
        self.game_board.update_state(p0, p1, self.card_db, legal_actions, god_view)

        pending = EngineCompat.get_pending_query(self.gs)
        if EngineCompat.is_waiting_for_user_input(self.gs) and pending is not None and getattr(pending, 'query_type', '') == "SELECT_TARGET":
            params = getattr(pending, 'params', {})
            min_targets = params.get('min', 1) if hasattr(params, 'get') else 1
            max_targets = params.get('max', 99) if hasattr(params, 'get') else 99

            current = len(self.input_handler.selected_targets)
            self.control_panel.set_confirm_button_text(f"{tr('Confirm')} ({current}/{min_targets}-{max_targets})")
            self.control_panel.set_confirm_button_visible(True)
            self.control_panel.set_confirm_button_enabled(current >= min_targets)

            self.game_board.set_selection_mode(self.input_handler.selected_targets)
        else:
            self.control_panel.set_confirm_button_visible(False)

def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    if dm_ai_module is None:
        QMessageBox.critical(None, tr("Error"),
            tr("dm_ai_module not found. Please build the C++ extension."))
        sys.exit(1)

    window = GameWindow()
    window.show()
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
