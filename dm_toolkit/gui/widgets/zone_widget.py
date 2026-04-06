# -*- coding: utf-8 -*-
try:
    from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout
    from PyQt6.QtCore import Qt, pyqtSignal
except Exception:
    # Provide lightweight shims for headless/testing environments
    class _DummySignal:
        def __init__(self, *args, **kwargs): pass
        def emit(self, *a, **k): return None

    class _DummyWidget:
        def __init__(self, *args, **kwargs): pass
        def setParent(self, *a, **k): pass
        def window(self): return None

    class _DummyLayout:
        def __init__(self, *a, **k): pass
        def setContentsMargins(self, *a, **k): pass
        def setAlignment(self, *a, **k): pass
        def setSpacing(self, *a, **k): pass
        def addWidget(self, *a, **k): pass
        def itemAt(self, i): return None

    class _DummyLabel:
        def __init__(self, text=""): self._text = text
        def setFixedWidth(self, *a, **k): pass
        def setWordWrap(self, *a, **k): pass
        def setAlignment(self, *a, **k): pass
        def setStyleSheet(self, *a, **k): pass
        def setVisible(self, *a, **k): pass

    class _DummyScrollArea:
        def __init__(self, *a, **k): pass
        def setWidgetResizable(self, *a, **k): pass
        def setMinimumHeight(self, *a, **k): pass
        def setVerticalScrollBarPolicy(self, *a, **k): pass
        def setWidget(self, *a, **k): pass

    QWidget = _DummyWidget
    QHBoxLayout = _DummyLayout
    QVBoxLayout = _DummyLayout
    QLabel = _DummyLabel
    QScrollArea = _DummyScrollArea
    Qt = type('X', (), {'AlignmentFlag': type('A', (), {'AlignCenter': 0}), 'ScrollBarPolicy': type('S', (), {'ScrollBarAlwaysOff': 0})})
    pyqtSignal = _DummySignal
from .card_widget import CardWidget
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.card_helpers import get_card_civilization, get_card_civilizations
# 再発防止: wrap_action は wrap_command の後方互換エイリアス。wrap_command を使用すること。
from dm_toolkit.commands import wrap_command
from dm_toolkit.gui.styles.civ_colors import CIV_ORB_COLORS, CIV_NAMES_JA
import logging
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect

# module logger
logger = logging.getLogger('dm_toolkit.gui.widgets.zone_widget')

# 文明オーブの表示順
_CIV_ORDER = ["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO", "COLORLESS"]


class ZoneWidget(QWidget):
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id
    command_triggered = pyqtSignal(object)  # CommandDef を emit（再発防止: 旧 action_triggered）
    card_double_clicked = pyqtSignal(int, int) # card_id, instance_id

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.cards = []
        self.legal_commands = []  # 再発防止: legal_actions から legal_commands に改名
        self._is_mana_zone = "Mana" in title or "マナ" in title
        self._card_icon_height = CardWidget.ICON_HEIGHT
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title Label (Vertical)
        self.title_label = QLabel(self.title.replace(" ", "\n"))
        if hasattr(self.title_label, 'setFixedWidth'):
            self.title_label.setFixedWidth(40)
        elif hasattr(self.title_label, 'setFixedSize'):
            try:
                self.title_label.setFixedSize(40, 100)
            except Exception:
                pass

        if hasattr(self.title_label, 'setWordWrap'):
            self.title_label.setWordWrap(True)

        if hasattr(self.title_label, 'setAlignment') and hasattr(Qt, 'AlignmentFlag'):
            try:
                self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass

        if hasattr(self.title_label, 'setStyleSheet'):
            try:
                self.title_label.setStyleSheet("background-color: #ddd; border: 1px solid #999; font-weight: bold;")
            except Exception:
                pass
        main_layout.addWidget(self.title_label)

        # マナゾーン専用: 右端に文明オーブパネルを配置
        if self._is_mana_zone:
            self._orb_panel = QWidget()
            self._orb_panel.setFixedWidth(52)
            self._orb_layout = QVBoxLayout(self._orb_panel)
            self._orb_layout.setContentsMargins(2, 2, 2, 2)
            self._orb_layout.setSpacing(1)
            self._orb_labels: dict = {}  # civ -> QLabel
        
        # Scroll Area for Cards
        self.scroll_area = QScrollArea()
        if hasattr(self.scroll_area, 'setWidgetResizable'):
            try:
                self.scroll_area.setWidgetResizable(True)
            except Exception:
                pass
        if hasattr(self.scroll_area, 'setMinimumHeight'):
            try:
                # 再発防止: カードアイコン高の変更時に行高だけ据え置かれると見切れ/余白過多になるため、
                # アイコン高を基準にスクロール領域の最小高を同期する。
                self.scroll_area.setMinimumHeight(self._card_icon_height + 14)
            except Exception:
                pass
        if hasattr(self.scroll_area, 'setVerticalScrollBarPolicy') and hasattr(Qt, 'ScrollBarPolicy'):
            try:
                self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except Exception:
                pass
        
        self.card_container = QWidget()
        self.card_layout = QHBoxLayout(self.card_container)
        if hasattr(self.card_layout, 'setAlignment') and hasattr(Qt, 'AlignmentFlag'):
            try:
                self.card_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            except Exception:
                pass
        if hasattr(self.card_layout, 'setContentsMargins'):
            try:
                self.card_layout.setContentsMargins(5, 5, 5, 5)
            except Exception:
                pass
        if hasattr(self.card_layout, 'setSpacing'):
            try:
                self.card_layout.setSpacing(5)
            except Exception:
                pass
        
        if hasattr(self.scroll_area, 'setWidget'):
            try:
                self.scroll_area.setWidget(self.card_container)
            except Exception:
                pass
        main_layout.addWidget(self.scroll_area)

        # マナゾーンはオーブパネルを右端に追加
        if self._is_mana_zone:
            main_layout.addWidget(self._orb_panel)

    def _update_mana_orbs(self, card_data_list, card_db):
        """マナゾーンの文明オーブを更新する。アンタップ(利用可能)とタップ済みの数を文明別に集計して表示。"""
        if not self._is_mana_zone:
            return
        try:
            # 文明別に (アンタップ数, タップ数) を集計
            counts: dict = {}  # civ -> [untapped, tapped]
            for c_data in card_data_list:
                cid = c_data.get('id', -1)
                tapped = c_data.get('tapped', False)
                if cid in card_db:
                    card_def = card_db[cid]
                    raw_civ = get_card_civilization(card_def)
                    civs = raw_civ if isinstance(raw_civ, list) else [raw_civ]
                    for civ in civs:
                        if civ not in counts:
                            counts[civ] = [0, 0]
                        if tapped:
                            counts[civ][1] += 1
                        else:
                            counts[civ][0] += 1

            # 既存のラベルをクリア
            for lbl in self._orb_labels.values():
                try:
                    lbl.setParent(None)
                except Exception:
                    pass
            self._orb_labels.clear()

            # 表示する文明を固定順でソート
            display_civs = [c for c in _CIV_ORDER if c in counts]
            for civ in counts:
                if civ not in display_civs:
                    display_civs.append(civ)

            for civ in display_civs:
                untapped, tapped_n = counts[civ]
                fill, border = CIV_ORB_COLORS.get(civ, ("#D3D3D3", "#808080"))
                name = CIV_NAMES_JA.get(civ, civ)
                # 利用可能数(アンタップ) / 合計数
                total = untapped + tapped_n
                txt = f"{name}\n{untapped}/{total}"
                lbl = QLabel(txt)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # 利用可能なし（全タップ）→暗くする
                if untapped == 0:
                    lbl.setStyleSheet(
                        f"background-color: {fill}55; color: #888; border: 1px solid {border}66;"
                        f" border-radius: 4px; font-size: 9px; font-weight: bold;"
                    )
                else:
                    lbl.setStyleSheet(
                        f"background-color: {fill}; color: white; border: 2px solid {border};"
                        f" border-radius: 4px; font-size: 9px; font-weight: bold;"
                        f" text-shadow: 1px 1px 2px black;"
                    )
                tooltip_parts = []
                if untapped > 0:
                    tooltip_parts.append(f"利用可能: {untapped}枚")
                if tapped_n > 0:
                    tooltip_parts.append(f"使用済み: {tapped_n}枚")
                lbl.setToolTip(f"{name} — " + " / ".join(tooltip_parts))
                self._orb_layout.addWidget(lbl)
                self._orb_labels[civ] = lbl

            if not display_civs:
                # マナゼロ時は空表示
                pass
        except Exception as e:
            logger.debug(f"[ManaOrbs] update failed: {e}")

    def set_legal_commands(self, commands: list) -> None:
        self.legal_commands = commands
        # Update existing widgets if possible, but usually update_cards handles recreation
        for widget in self.cards:
             if widget.instance_id != -1:
                 relevant = [c for c in self.legal_commands if getattr(c, 'source_instance_id', -1) == widget.instance_id]
                 widget.update_legal_commands(relevant)

    # 後方互換エイリアス
    def set_legal_actions(self, actions: list) -> None:
        self.set_legal_commands(actions)

    def update_cards(self, card_data_list, card_db, civ_map=None, legal_commands=None, collapsed=None):
        # Update cached legal commands if provided
        if legal_commands is not None:
            self.legal_commands = legal_commands

        # Save necessary data for potential popup
        self.card_db = card_db
        self.last_card_data_list = card_data_list
        self.civ_map = civ_map

        # Debug: Log card_db type and card_data_list info
        if hasattr(card_data_list, '__len__') and len(card_data_list) > 0:
            first_card = card_data_list[0] if card_data_list else {}
            card_id = first_card.get('id', -1)
            logger.debug(f"[zone {self.title}] card_data_list has {len(card_data_list)} cards, first id={card_id}, card_db type={type(card_db)}")

        # If popup is active, update it too
        if hasattr(self, 'active_popup') and self.active_popup and self.active_popup.isVisible():
            self.active_popup.update_content(card_data_list, card_db, civ_map, legal_commands)

        # Clear existing
        for i in reversed(range(self.card_layout.count())):
            item = self.card_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        self.cards = []

        # Check for Bundle Visualization
        is_deck = "Deck" in self.title or "デッキ" in self.title
        is_shield = "Shield" in self.title or "シールド" in self.title
        is_mana = "Mana" in self.title or "マナ" in self.title
        is_grave = "Graveyard" in self.title or "墓地" in self.title

        # Determine collapsed state
        if collapsed is None:
            # Default behavior
            if is_deck or is_shield or is_mana or is_grave:
                collapsed = True
            else:
                collapsed = False

        if collapsed and card_data_list:
            # Single Bundle Representation
            count = len(card_data_list)
            display_name = ""
            is_face_down = True
            card_id_display = 0
            civ_display = "COLORLESS"

            # Use ID 0 (Back of Card) or Generic
            if is_deck:
                display_name = tr("Deck ({count})").format(count=count)
            elif is_shield:
                display_name = tr("Shield ({count})").format(count=count)
            elif is_mana:
                display_name = tr("Mana ({count})").format(count=count)
                is_face_down = False # Mana is public usually, but bundle is symbolic

                # Try to show the top card if available
                if count > 0:
                    top_card = card_data_list[-1]
                    tid = top_card['id']
                    if tid in card_db:
                        card_def = card_db[tid]
                        pass
                # 文明オーブを更新（利用可能文明の表示）
                self._update_mana_orbs(card_data_list, card_db)

            elif is_grave:
                display_name = tr("Graveyard ({count})").format(count=count)
                is_face_down = False
                # Show top card of graveyard
                if count > 0:
                    top_card = card_data_list[-1]
                    tid = top_card['id']
                    card_id_display = tid
                    if tid in card_db:
                         card_def = card_db[tid]
                         # Handle both dict and object card definitions
                         card_name = card_def['name'] if isinstance(card_def, dict) else card_def.name
                         display_name = f"{card_name}\n({tr('Graveyard')}: {count})"
                         civ_display = get_card_civilizations(card_def)


            widget = CardWidget(card_id_display, display_name, 0, 0, civ_display, False, -1, None, is_face_down)

            # Clicking behavior
            if is_mana or is_grave:
                # Open Popup
                safe_connect(widget, 'clicked', self._open_popup)
            else:
                # Standard emit
                safe_connect(widget, 'clicked', lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))

            safe_connect(widget, 'hovered', self.card_hovered.emit)
            self.card_layout.addWidget(widget)
            self.cards.append(widget)
            return

        # Normal Visualization
        for c_data in card_data_list:
            # c_data: (card_id, is_tapped) or just card_id
            cid = c_data['id']
            tapped = c_data.get('tapped', False)
            instance_id = c_data.get('instance_id', -1)
            
            # Filter actions for this card
            relevant_actions = []
            if instance_id != -1:
                for a in self.legal_commands:
                    # Support both dict and object command representations
                    if hasattr(a, 'source_instance_id'):
                        if a.source_instance_id == instance_id:
                            relevant_actions.append(a)
                    elif isinstance(a, dict):
                        if a.get('source_instance_id') == instance_id or a.get('instance_id') == instance_id:
                            relevant_actions.append(a)
                    else:
                        # Try to_dict() method
                        try:
                            d = a.to_dict()
                            if d.get('source_instance_id') == instance_id or d.get('instance_id') == instance_id:
                                relevant_actions.append(a)
                        except:
                            pass

            if cid in card_db:
                card_def = card_db[cid]
                civ = get_card_civilizations(card_def)
                
                # Support both dict and object formats for card_def
                card_name = card_def['name'] if isinstance(card_def, dict) else card_def.name
                card_cost = card_def['cost'] if isinstance(card_def, dict) else card_def.cost
                card_power = card_def['power'] if isinstance(card_def, dict) else card_def.power
                
                widget = CardWidget(
                    cid, card_name, card_cost, card_power, 
                    civ, tapped, instance_id,
                    legal_commands=relevant_actions
                )
                safe_connect(widget, 'clicked', lambda i_id, c_id=cid: self.card_clicked.emit(c_id, i_id))
                safe_connect(widget, 'hovered', self.card_hovered.emit)
                safe_connect(widget, 'command_triggered', self._handle_command_triggered)
                safe_connect(widget, 'double_clicked', lambda i_id, c_id=cid: self.card_double_clicked.emit(c_id, i_id))
                self.card_layout.addWidget(widget)
                self.cards.append(widget)
            else:
                # Unknown/Masked
                # Pass is_face_down=True
                widget = CardWidget(0, "???", 0, 0, "COLORLESS", False, instance_id, None, True, legal_commands=relevant_actions)
                safe_connect(widget, 'clicked', lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
                safe_connect(widget, 'hovered', self.card_hovered.emit)
                safe_connect(widget, 'command_triggered', self._handle_command_triggered)
                safe_connect(widget, 'double_clicked', lambda i_id, c_id=0: self.card_double_clicked.emit(c_id, i_id))
                self.card_layout.addWidget(widget)

        # ノーマル表示（展開時）でも文明オーブを更新
        if self._is_mana_zone:
            self._update_mana_orbs(card_data_list, card_db)

    def set_card_selected(self, instance_id, selected):
        for widget in self.cards:
            if widget.instance_id == instance_id:
                widget.set_selected(selected)
                return

    def highlight_cards(self, instance_ids: list, mode: str = "legal"):
        """指定 instance_id のカードウィジェットをハイライトする（方針A）。
        mode: "legal" → 緑枠（操作可能）, "target" → 黄枠（有効対象）
        """
        for widget in self.cards:
            if widget.instance_id in instance_ids:
                if mode == "target":
                    widget.set_highlight_target(True)
                else:
                    widget.set_highlight_legal(True)

    def clear_highlights(self):
        """全カードウィジェットのハイライトをリセットする（方針A）。"""
        for widget in self.cards:
            widget.clear_highlights()

    def _handle_command_triggered(self, cmd):
        """
        コマンドトリガーシグナルをインターセプトし、上位にバブルアップする前に wrap_command でラップする。
        """
        # 再発防止: wrap_action は wrap_command に改名済み。
        wrapped = wrap_command(cmd)
        self.command_triggered.emit(wrapped)

    def _open_popup(self, *args):
        from dm_toolkit.gui.widgets.zone_popup import ZonePopup
        # Pass civ_map if available
        civ_map = getattr(self, 'civ_map', None)
        popup = ZonePopup(self.title, self.last_card_data_list, self.card_db, self.civ_map, self.legal_commands, parent=self.window())

        # Track active popup to update it if game state changes while it's open
        self.active_popup = popup

        # Connect signals from popup's inner widget to our signals
        # So if user clicks a card in popup, it behaves as if they clicked it in the zone
        from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
        safe_connect(popup.zone_widget, 'card_clicked', self.card_clicked.emit)
        safe_connect(popup.zone_widget, 'card_double_clicked', self.card_double_clicked.emit)
        safe_connect(popup.zone_widget, 'command_triggered', self.command_triggered.emit)
        safe_connect(popup.zone_widget, 'card_hovered', self.card_hovered.emit)

        popup.exec()

        self.active_popup = None
