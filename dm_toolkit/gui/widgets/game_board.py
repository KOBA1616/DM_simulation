# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QFrame, QPushButton
)
from PyQt6.QtCore import pyqtSignal, Qt

from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.i18n import tr

class GameBoard(QWidget):
    """
    Manages the game board layout including player zones.
    """
    command_triggered = pyqtSignal(object)  # CommandDef を emit（再発防止: 旧 action_triggered）
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_double_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)

        # 方針C: 状態通知バナー（ボード最上部・固定高さ・常時表示でレイアウト移動防止）
        self.action_hint_label = QLabel("")
        self.action_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_hint_label.setWordWrap(False)
        self.action_hint_label.setFixedHeight(26)
        self._banner_active_style = (
            "background-color: #1a1a2e; color: #e0e0e0; font-size: 12px; "
            "font-weight: bold; padding: 2px 8px; border-bottom: 1px solid #444;"
        )
        self._banner_inactive_style = (
            "background-color: transparent; color: transparent; border: none;"
        )
        self.action_hint_label.setStyleSheet(self._banner_inactive_style)
        # 再発防止: setVisible()でレイアウトが動くため、常時表示してスタイルで切り替える
        self.layout_main.addWidget(self.action_hint_label)

        # P1 Zones (Opponent)
        self.p1_zones = QWidget()
        self.p1_layout = QVBoxLayout(self.p1_zones)
        self.p1_hand = ZoneWidget(tr("P1 Hand"))
        self.p1_mana = ZoneWidget(tr("P1 Mana"))
        self.p1_graveyard = ZoneWidget(tr("P1 Graveyard"))
        self.p1_battle = ZoneWidget(tr("P1 Battle Zone"))
        self.p1_shield = ZoneWidget(tr("P1 Shield Zone"))
        self.p1_deck_zone = ZoneWidget(tr("P1 Deck"))

        self.p1_layout.addWidget(self.p1_hand)
        p1_row2 = QHBoxLayout()
        p1_row2.addWidget(self.p1_mana, stretch=3)
        p1_row2.addWidget(self.p1_shield, stretch=2)
        p1_row2.addWidget(self.p1_graveyard, stretch=1)
        self.p1_layout.addLayout(p1_row2)

        p1_battle_row = QHBoxLayout()
        p1_battle_row.addWidget(self.p1_battle, stretch=5)
        p1_battle_row.addWidget(self.p1_deck_zone, stretch=1)
        self.p1_layout.addLayout(p1_battle_row)

        # P0 Zones (Self)
        self.p0_zones = QWidget()
        self.p0_layout = QVBoxLayout(self.p0_zones)
        self.p0_battle = ZoneWidget(tr("P0 Battle Zone"))
        self.p0_deck_zone = ZoneWidget(tr("P0 Deck"))
        self.p0_shield = ZoneWidget(tr("P0 Shield Zone"))
        self.p0_mana = ZoneWidget(tr("P0 Mana"))
        self.p0_graveyard = ZoneWidget(tr("P0 Graveyard"))
        self.p0_hand = ZoneWidget(tr("P0 Hand"))

        p0_battle_row = QHBoxLayout()
        p0_battle_row.addWidget(self.p0_battle, stretch=5)
        p0_battle_row.addWidget(self.p0_deck_zone, stretch=1)
        self.p0_layout.addLayout(p0_battle_row)

        p0_row2 = QHBoxLayout()
        p0_row2.addWidget(self.p0_mana, stretch=3)
        p0_row2.addWidget(self.p0_shield, stretch=2)
        p0_row2.addWidget(self.p0_graveyard, stretch=1)
        self.p0_layout.addLayout(p0_row2)
        self.p0_layout.addWidget(self.p0_hand)

        # Connect Signals
        self._connect_zone(self.p0_hand)
        self._connect_zone(self.p0_mana)
        self._connect_zone(self.p0_battle)
        self._connect_zone(self.p0_graveyard)
        self._connect_zone(self.p0_shield) # Shield usually not clickable for actions but maybe hover

        # Opponent zones usually just hover
        for z in [self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard]:
            z.card_hovered.connect(self.card_hovered.emit)

        # Splitter
        self.board_splitter = QSplitter(Qt.Orientation.Vertical)
        self.board_splitter.addWidget(self.p1_zones)
        self.board_splitter.addWidget(self.p0_zones)

        # 再発防止: ActionPanel 右サイドバーは control_panel.py の P0 操作セクションに移行済み
        self.layout_main.addWidget(self.board_splitter)

        # 方針D: フローティング確定ボタン（ボード上に重ねる絶対位置配置）
        self.floating_confirm = QFrame(self)
        self.floating_confirm.setFrameShape(QFrame.Shape.StyledPanel)
        self.floating_confirm.setStyleSheet(
            "background-color: rgba(39,174,96,220); border-radius: 8px; "
            "border: 2px solid #1e8449;"
        )
        self.floating_confirm.setFixedSize(180, 44)
        fc_layout = QHBoxLayout(self.floating_confirm)
        fc_layout.setContentsMargins(6, 4, 6, 4)
        self.floating_confirm_btn = QPushButton(tr("Confirm Selection"))
        self.floating_confirm_btn.setStyleSheet(
            "background: transparent; color: white; font-weight: bold; "
            "font-size: 12px; border: none;"
        )
        fc_layout.addWidget(self.floating_confirm_btn)
        self.floating_confirm.setVisible(False)
        # 再発防止: resizeEvent でフローティングボタンの位置を更新すること

    def _connect_zone(self, zone: ZoneWidget):
        zone.command_triggered.connect(self.command_triggered.emit)
        zone.card_clicked.connect(self.card_clicked.emit)
        zone.card_double_clicked.connect(self.card_double_clicked.emit)
        zone.card_hovered.connect(self.card_hovered.emit)

    def resizeEvent(self, event):
        """方針D: リサイズ時にフローティング確定ボタンを右下に再配置する。"""
        super().resizeEvent(event)
        if hasattr(self, 'floating_confirm'):
            margin = 16
            x = self.width() - self.floating_confirm.width() - margin
            y = self.height() - self.floating_confirm.height() - margin
            self.floating_confirm.move(x, y)

    # ---- 方針C: 状態通知バナー ----
    def set_action_hint(self, msg: str):
        """ボード上部バナーにヒントメッセージを設定する（方針C）。
        再発防止: setVisible()はレイアウト移動の原因になるため使わない。
                  テキストが空の場合は透明スタイルに切り替えて高さを維持する。
        """
        if msg:
            self.action_hint_label.setText(msg)
            self.action_hint_label.setStyleSheet(self._banner_active_style)
        else:
            self.action_hint_label.setText("")
            self.action_hint_label.setStyleSheet(self._banner_inactive_style)

    # ---- 方針A: ハイライト ----
    def _all_zones(self) -> list:
        return [
            self.p0_hand, self.p0_mana, self.p0_battle, self.p0_shield, self.p0_graveyard,
            self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard,
        ]

    def clear_highlights(self):
        """全ゾーンの全カードハイライトをリセットする（方針A）。"""
        for zone in self._all_zones():
            zone.clear_highlights()

    def highlight_legal_commands(self, legal_cmds: list):
        """人間プレイヤーが操作可能なカードを緑枠ハイライトする（方針A）。"""
        self.clear_highlights()
        legal_ids = set()
        for cmd in legal_cmds:
            try:
                d = cmd.to_dict()
            except Exception:
                d = {}
            iid = d.get('instance_id') or d.get('source_instance_id')
            if iid is not None:
                legal_ids.add(iid)
        if legal_ids:
            for zone in [self.p0_hand, self.p0_battle, self.p0_mana]:
                zone.highlight_cards(legal_ids, mode="legal")

    def highlight_valid_targets(self, valid_targets: list):
        """SELECT_TARGET 時に有効対象カードを黄枠ハイライトする（方針A）。"""
        self.clear_highlights()
        target_ids = set(valid_targets)
        if target_ids:
            for zone in self._all_zones():
                zone.highlight_cards(target_ids, mode="target")

    # ---- 方針D: フローティング確定ボタン ----
    def set_floating_confirm(self, visible: bool, text: str = ""):
        """フローティング確定ボタンの表示・テキストを更新する（方針D）。"""
        if text:
            self.floating_confirm_btn.setText(text)
        self.floating_confirm.setVisible(visible)
        if visible:
            self.floating_confirm.raise_()  # 最前面に移動

    def update_state(self, p0_data: Any, p1_data: Any, card_db: Any, legal_commands: List[Any], god_view: bool = False):
        """
        Updates all zones based on player data objects (EngineCompat.get_player result).
        """
        # Convert card_db to dict if it's a list
        if isinstance(card_db, list):
            card_db_dict = {card['id']: card for card in card_db}
        else:
            card_db_dict = card_db
        
        def convert_zone(zone_cards: List[Any], hide: bool=False) -> List[Dict[str, Any]]:
            if hide: return [{'id': -1, 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]
            return [{'id': getattr(c, 'card_id', -1), 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]

        # P0 Update (Self)
        self.p0_hand.update_cards(convert_zone(p0_data.hand), card_db_dict, legal_commands=legal_commands)
        self.p0_mana.update_cards(convert_zone(p0_data.mana_zone), card_db_dict, legal_commands=legal_commands)
        self.p0_battle.update_cards(convert_zone(p0_data.battle_zone), card_db_dict, legal_commands=legal_commands)
        self.p0_shield.update_cards(convert_zone(p0_data.shield_zone), card_db_dict, legal_commands=legal_commands)
        self.p0_graveyard.update_cards(convert_zone(p0_data.graveyard), card_db_dict, legal_commands=legal_commands)
        self.p0_deck_zone.update_cards(convert_zone(p0_data.deck, hide=True), card_db_dict, legal_commands=legal_commands)

        # P1 Update (Opponent)
        self.p1_hand.update_cards(convert_zone(p1_data.hand, hide=not god_view), card_db_dict)
        self.p1_mana.update_cards(convert_zone(p1_data.mana_zone), card_db_dict)
        self.p1_battle.update_cards(convert_zone(p1_data.battle_zone), card_db_dict)
        self.p1_shield.update_cards(convert_zone(p1_data.shield_zone, hide=not god_view), card_db_dict)
        self.p1_graveyard.update_cards(convert_zone(p1_data.graveyard), card_db_dict)
        self.p1_deck_zone.update_cards(convert_zone(p1_data.deck, hide=True), card_db_dict)

    def set_selection_mode(self, selected_targets: List[int]):
        """
        Highlights selected cards across zones.
        """
        zones = [
            self.p0_hand, self.p0_mana, self.p0_battle, self.p0_shield, self.p0_graveyard,
            self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard
        ]
        for zone in zones:
            for target_id in selected_targets:
                zone.set_card_selected(target_id, True)
