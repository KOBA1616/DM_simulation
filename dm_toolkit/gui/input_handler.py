# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from PyQt6.QtWidgets import QMessageBox

from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command
from dm_toolkit.gui.dialogs.selection_dialog import CardSelectionDialog

from dm_toolkit import commands
# Prefer the command-first wrapper
generate_legal_commands = commands.generate_legal_commands

if TYPE_CHECKING:
    from dm_toolkit.gui.app import GameWindow
    from dm_toolkit.gui.game_session import GameSession
    from dm_toolkit.dm_types import GameState, CardDB


class GameInputHandler:
    def __init__(self, window: 'GameWindow', session: 'GameSession'):
        self.window = window
        self.session = session
        self.selected_targets: List[int] = []

        # 再発防止: ActionPanel は control_panel に移行済み
        try:
            self.window.control_panel.action_command_selected.connect(
                self._on_panel_command_selected
            )
        except Exception:
            pass

    @property
    def gs(self) -> 'GameState':
        return self.session.gs

    @property
    def card_db(self) -> 'CardDB':
        # Prefer native_card_db for command generation if available
        # Check for existence, not truthiness (CardDatabase might be empty but still valid)
        if hasattr(self.window, 'native_card_db') and self.window.native_card_db is not None:
            return self.window.native_card_db
        return self.window.card_db

    def on_card_clicked(self, card_id: int, instance_id: int) -> None:
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.window.control_panel.is_p0_human():
            return

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

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
            if d.get('instance_id') == instance_id or d.get('source_instance_id') == instance_id:
                relevant_cmds.append(c)

        if not relevant_cmds:
            # 関連コマンドがない場合はパネルをクリア
            try:
                self.window.control_panel.clear_action_commands()
            except Exception:
                pass
            return

        # 再発防止: シングルクリックは常に P0 操作パネルに表示して確認させる（誤操作防止）。
        try:
            self.window.control_panel.set_action_commands(
                relevant_cmds, self.gs, self.card_db
            )
        except Exception:
            # フォールバック: control_panel が利用不可の場合のみ即実行
            if len(relevant_cmds) == 1:
                self.session.execute_command(relevant_cmds[0])

    def on_card_double_clicked(self, card_id: int, instance_id: int) -> None:
        """Handle double-click to quickly play the most common action (Play or Mana Charge)."""
        if EngineCompat.get_active_player_id(self.gs) != 0 or not self.window.control_panel.is_p0_human():
            return

        if EngineCompat.is_waiting_for_user_input(self.gs):
            return

        cmds = generate_legal_commands(self.gs, self.card_db)
        relevant_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
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
            if cmd_type == 'PLAY_FROM_ZONE' or cmd_type == 'PLAY_CARD':
                play_cmd = cmd
            elif cmd_type == 'MANA_CHARGE':
                mana_cmd = cmd
            elif cmd_type == 'ATTACK' or cmd_type == 'ATTACK_CREATURE' or cmd_type == 'ATTACK_PLAYER':
                attack_cmd = cmd
            elif not other_cmd:
                other_cmd = cmd

        if play_cmd:
            self.session.execute_command(play_cmd)
        elif attack_cmd:
            self.session.execute_command(attack_cmd)
        elif other_cmd:
            self.session.execute_command(other_cmd)
        elif mana_cmd:
            self.session.execute_command(mana_cmd)

    def handle_user_input_request(self) -> None:
        """Called by GameSession when input is needed."""
        # Stop simulation if running
        if self.window.is_running:
            self.window.timer.stop()
            self.window.control_panel.set_start_button_text(tr("Start Sim"))
            self.window.is_running = False

        query = EngineCompat.get_pending_query(self.gs)
        if query.query_type == "SELECT_OPTION":
            from PyQt6.QtWidgets import QInputDialog
            options = query.options
            item, ok = QInputDialog.getItem(self.window, tr("Select Option"), tr("Choose an option:"), options, 0, False)
            if ok and item:
                idx = options.index(item)
                self.session.resume_from_input(idx)
        elif query.query_type == "SELECT_TARGET":
            valid_targets = query.valid_targets
            if not valid_targets:
                return

            # 方針A: 有効対象を黄枠ハイライト
            try:
                self.window.game_board.highlight_valid_targets(list(valid_targets))
            except Exception:
                pass

            # 方針B/C: SELECT_TARGET 時はパネルを待機モードにし、バナーでガイド
            params = getattr(query, 'params', {})
            min_sel = params.get('min', 1) if hasattr(params, 'get') else 1
            max_sel = params.get('max', 99) if hasattr(params, 'get') else 99
            hint_msg = tr("Select target ({min}-{max})").format(min=min_sel, max=max_sel)
            try:
                self.window.game_board.set_action_hint(hint_msg)
                self.window.control_panel.set_action_waiting_mode(
                    True, tr("Click targets on the board,\nthen press Confirm")
                )
            except Exception:
                pass

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
                    if found:
                        items.append(found)
                min_sel = query.params.get('min', 1)
                max_sel = query.params.get('max', 99)
                dialog = CardSelectionDialog(tr("Select Cards"), tr("Please select cards:"), items, min_sel, max_sel, self.window, self.card_db)
                if dialog.exec():
                    indices = dialog.get_selected_indices()
                    selected_instance_ids = [items[i].instance_id for i in indices]
                    self.session.resume_from_input(cast(Any, selected_instance_ids))
                return
            else:
                # Targets are on board/hand/mana, highlight them (update_ui also calls highlight_valid_targets)
                self.window.update_ui()

    def confirm_selection(self) -> None:
        if not EngineCompat.is_waiting_for_user_input(self.gs):
            return
        query = EngineCompat.get_pending_query(self.gs)
        min_targets = query.params.get('min', 1)
        if len(self.selected_targets) < min_targets:
            msg = tr("Please select at least {min_targets} target(s).").format(min_targets=min_targets)
            QMessageBox.warning(self.window, tr("Invalid Selection"), msg)
            return
        targets = list(self.selected_targets)
        self.selected_targets = []
        self.window.control_panel.set_confirm_button_visible(False)
        # 方針C+D: バナーとフローティングボタンをリセット
        try:
            self.window.game_board.set_action_hint("")
            self.window.game_board.set_floating_confirm(False)
            self.window.control_panel.set_action_waiting_mode(False)
        except Exception:
            pass
        self.session.resume_from_input(targets)

    def _on_panel_command_selected(self, cmd: Any) -> None:
        """ActionPanel から選択されたコマンドを実行する（方針B）。"""
        # 再発防止: パネル選択後はコマンドを即実行。ハイライトクリアは update_ui が担当。
        try:
            self.window.game_board.clear_highlights()
        except Exception:
            pass
        self.session.execute_command(cmd)

    def on_resolve_effect_from_stack(self, index: int) -> None:
        """Resolve a pending effect by its vector index.
        
        Args:
            index: Vector index in the pending_effects array
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[InputHandler] on_resolve_effect_from_stack called with index {index}")

        cmds = generate_legal_commands(self.gs, self.card_db)
        resolve_cmds = []
        for c in cmds:
            try:
                d = c.to_dict()
            except Exception:
                d = {}
            if d.get('type') == 'RESOLVE_EFFECT':
                resolve_cmds.append((c, d))

        logger.debug(f"[InputHandler] Found {len(resolve_cmds)} RESOLVE_EFFECT commands")
        target_cmd = None
        for c, d in resolve_cmds:
            logger.debug(f"[InputHandler] Checking RESOLVE_EFFECT with slot_index {d.get('slot_index')}")
            if d.get('slot_index') == index:
                target_cmd = c
                logger.debug(f"[InputHandler] Found matching RESOLVE_EFFECT command")
                break
        if target_cmd:
            logger.debug(f"[InputHandler] Executing RESOLVE_EFFECT command")
            self.session.execute_command(target_cmd)
        elif len(resolve_cmds) == 1:
            logger.debug(f"[InputHandler] Only one RESOLVE_EFFECT, executing it")
            self.session.execute_command(resolve_cmds[0][0])
        else:
            logger.warning(f"[InputHandler] No matching RESOLVE_EFFECT command found for index {index}")
