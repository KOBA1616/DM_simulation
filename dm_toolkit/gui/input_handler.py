# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.localization import tr, describe_command
from dm_toolkit.gui.dialogs.selection_dialog import CardSelectionDialog

if TYPE_CHECKING:
    from dm_toolkit.gui.app import GameWindow
    from dm_toolkit.gui.game_session import GameSession
    from dm_toolkit.types import GameState, CardDB

class GameInputHandler:
    def __init__(self, window: 'GameWindow', session: 'GameSession'):
        self.window = window
        self.session = session
        self.selected_targets: List[int] = []

    @property
    def gs(self) -> 'GameState':
        return self.session.gs

    @property
    def card_db(self) -> 'CardDB':
        return self.window.card_db

    def on_card_clicked(self, card_id: int, instance_id: int) -> None:
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.window.control_panel.is_p0_human(): return

        if EngineCompat.is_waiting_for_user_input(self.gs):
             pending = EngineCompat.get_pending_query(self.gs)
             if pending.query_type == "SELECT_TARGET":
                 valid_targets = pending.valid_targets
                 if instance_id in valid_targets:
                     if instance_id in self.selected_targets:
                         self.selected_targets.remove(instance_id)
                     else:
                         query_max = pending.params.get('max', 99)
                         if len(self.selected_targets) < query_max:
                             self.selected_targets.append(instance_id)
                         else:
                             return
                     self.window.update_ui()
             return

        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try: d = c.to_dict()
            except: d = {}
            if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
                relevant_cmds.append(c)

        if not relevant_cmds: return

        if len(relevant_cmds) > 1:
            items = []
            for cmd in relevant_cmds:
                d = cmd.to_dict()
                desc = describe_command(d, self.gs, self.card_db)
                items.append({'description': desc, 'command': cmd})

            options = [item['description'] for item in items]
            item, ok = QInputDialog.getItem(self.window, tr("Select Action"), tr("Choose action to perform:"), options, 0, False)
            if ok and item:
                idx = options.index(item)
                self.session.execute_action(items[idx]['command'])
            return

        self.session.execute_action(relevant_cmds[0])

    def on_card_double_clicked(self, card_id: int, instance_id: int) -> None:
        """Handle double-click to quickly play the most common action (Play or Mana Charge)."""
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.window.control_panel.is_p0_human():
            return

        if EngineCompat.is_waiting_for_user_input(self.gs):
            return

        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try: d = c.to_dict()
            except: d = {}
            if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
                relevant_cmds.append((c, d))

        if not relevant_cmds:
            return

        play_cmd = None
        mana_cmd = None
        attack_cmd = None
        other_cmd = None

        for cmd, d in relevant_cmds:
            cmd_type = d.get('type', '')
            if cmd_type == 'PLAY_CARD':
                play_cmd = cmd
            elif cmd_type == 'MANA_CHARGE':
                mana_cmd = cmd
            elif cmd_type == 'ATTACK':
                attack_cmd = cmd
            elif not other_cmd:
                other_cmd = cmd

        if play_cmd: self.session.execute_action(play_cmd)
        elif attack_cmd: self.session.execute_action(attack_cmd)
        elif other_cmd: self.session.execute_action(other_cmd)
        elif mana_cmd: self.session.execute_action(mana_cmd)

    def handle_user_input_request(self) -> None:
        """Called by GameSession when input is needed."""
        # Stop simulation if running
        if self.window.is_running:
            self.window.timer.stop()
            self.window.control_panel.set_start_button_text(tr("Start Sim"))
            self.window.is_running = False

        query = EngineCompat.get_pending_query(self.gs)
        if query.query_type == "SELECT_OPTION":
             options = query.options
             item, ok = QInputDialog.getItem(self.window, tr("Select Option"), tr("Choose an option:"), options, 0, False)
             if ok and item:
                 idx = options.index(item)
                 self.session.resume_from_input(idx)
        elif query.query_type == "SELECT_TARGET":
             valid_targets = query.valid_targets
             if not valid_targets: return

             # Check if targets are in buffer (temp zone)
             first_target_id = valid_targets[0]
             in_buffer = False
             buffer_cards = EngineCompat.get_effect_buffer(self.gs)
             for c in buffer_cards:
                 if c.instance_id == first_target_id:
                     in_buffer = True
                     break

             if in_buffer:
                 items = []
                 for tid in valid_targets:
                     found = next((c for c in buffer_cards if c.instance_id == tid), None)
                     if found: items.append(found)
                 min_sel = query.params.get('min', 1)
                 max_sel = query.params.get('max', 99)
                 dialog = CardSelectionDialog(tr("Select Cards"), tr("Please select cards:"), items, min_sel, max_sel, self.window, self.card_db)
                 if dialog.exec():
                     indices = dialog.get_selected_indices()
                     selected_instance_ids = [items[i].instance_id for i in indices]
                     self.session.resume_from_input(cast(Any, selected_instance_ids))
                 return
             else:
                 # Targets are on board/hand/mana, highlight them
                 self.window.update_ui()

    def confirm_selection(self) -> None:
        if not EngineCompat.is_waiting_for_user_input(self.gs): return
        query = EngineCompat.get_pending_query(self.gs)
        min_targets = query.params.get('min', 1)
        if len(self.selected_targets) < min_targets:
            msg = tr("Please select at least {min_targets} target(s).").format(min_targets=min_targets)
            QMessageBox.warning(self.window, tr("Invalid Selection"), msg)
            return
        targets = list(self.selected_targets)
        self.selected_targets = []
        self.window.control_panel.set_confirm_button_visible(False)
        self.session.resume_from_input(targets)

    def on_resolve_effect_from_stack(self, index: int) -> None:
        from dm_toolkit.commands import generate_legal_commands

        cmds = generate_legal_commands(self.gs, self.card_db)
        resolve_cmds = []
        for c in cmds:
            try: d = c.to_dict()
            except: d = {}
            if d.get('type') == 'RESOLVE_EFFECT':
                resolve_cmds.append((c, d))

        target_cmd = None
        for c, d in resolve_cmds:
            if d.get('slot_index') == index:
                target_cmd = c
                break
        if target_cmd:
            self.session.execute_action(target_cmd)
        elif len(resolve_cmds) == 1:
            self.session.execute_action(resolve_cmds[0][0])
