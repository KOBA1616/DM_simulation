# -*- coding: utf-8 -*-
try:
    from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QScrollArea
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
    QLabel = _DummyLabel
    QScrollArea = _DummyScrollArea
    Qt = type('X', (), {'AlignmentFlag': type('A', (), {'AlignCenter': 0}), 'ScrollBarPolicy': type('S', (), {'ScrollBarAlwaysOff': 0})})
    pyqtSignal = _DummySignal
from .card_widget import CardWidget
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.card_helpers import get_card_civilization
from dm_toolkit.commands import wrap_action
import logging

# module logger
logger = logging.getLogger('dm_toolkit.gui.widgets.zone_widget')

class ZoneWidget(QWidget):
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id
    action_triggered = pyqtSignal(object) # Action object (or Command)
    card_double_clicked = pyqtSignal(int, int) # card_id, instance_id

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.cards = []
        self.legal_actions = []
        
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
        
        # Scroll Area for Cards
        self.scroll_area = QScrollArea()
        if hasattr(self.scroll_area, 'setWidgetResizable'):
            try:
                self.scroll_area.setWidgetResizable(True)
            except Exception:
                pass
        if hasattr(self.scroll_area, 'setMinimumHeight'):
            try:
                self.scroll_area.setMinimumHeight(150)
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

    def set_legal_actions(self, actions):
        self.legal_actions = actions
        # Update existing widgets if possible, but usually update_cards handles recreation
        # If we want live updates without recreation:
        for widget in self.cards:
             if widget.instance_id != -1:
                 relevant = [a for a in self.legal_actions if getattr(a, 'source_instance_id', -1) == widget.instance_id]
                 widget.update_legal_actions(relevant)

    def update_cards(self, card_data_list, card_db, civ_map=None, legal_actions=None, collapsed=None):
        # Update cached legal actions if provided
        if legal_actions is not None:
            self.legal_actions = legal_actions

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
            self.active_popup.update_content(card_data_list, card_db, civ_map, legal_actions)

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
                        # For bundle, we might want just "Mana (N)" text, but let's try to mimic "top card visible" if desired.
                        # However, bundling implies we don't see the list.
                        # If we just show the card back or generic info, it's safer.
                        # But let's check civ distribution?
                        # For now, generic "Mana (N)" is fine.
                        pass

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
                         display_name = f"{card_def.name}\n({tr('Graveyard')}: {count})"
                         civ_display = get_card_civilization(card_def)


            widget = CardWidget(card_id_display, display_name, 0, 0, civ_display, False, -1, None, is_face_down)

            # Clicking behavior
            if is_mana or is_grave:
                 # Open Popup
                 widget.clicked.connect(self._open_popup)
            else:
                 # Standard emit
                 widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))

            widget.hovered.connect(self.card_hovered.emit)
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
                for a in self.legal_actions:
                    # Support both dict and object action representations
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
                civ = get_card_civilization(card_def)
                
                # Support both dict and object formats for card_def
                card_name = card_def['name'] if isinstance(card_def, dict) else card_def.name
                card_cost = card_def['cost'] if isinstance(card_def, dict) else card_def.cost
                card_power = card_def['power'] if isinstance(card_def, dict) else card_def.power
                
                widget = CardWidget(
                    cid, card_name, card_cost, card_power, 
                    civ, tapped, instance_id,
                    legal_actions=relevant_actions
                )
                widget.clicked.connect(lambda i_id, c_id=cid: self.card_clicked.emit(c_id, i_id))
                widget.hovered.connect(self.card_hovered.emit)
                widget.action_triggered.connect(self._handle_action_triggered)
                widget.double_clicked.connect(lambda i_id, c_id=cid: self.card_double_clicked.emit(c_id, i_id))
                self.card_layout.addWidget(widget)
                self.cards.append(widget)
            else:
                # Unknown/Masked
                # Pass is_face_down=True
                widget = CardWidget(0, "???", 0, 0, "COLORLESS", False, instance_id, None, True, legal_actions=relevant_actions)
                widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
                widget.hovered.connect(self.card_hovered.emit)
                widget.action_triggered.connect(self._handle_action_triggered)
                widget.double_clicked.connect(lambda i_id, c_id=0: self.card_double_clicked.emit(c_id, i_id))
                self.card_layout.addWidget(widget)

    def set_card_selected(self, instance_id, selected):
        for widget in self.cards:
            if widget.instance_id == instance_id:
                widget.set_selected(selected)
                return

    def _handle_action_triggered(self, action):
        """
        Intercepts action triggered signal to potentially wrap it as a Command
        before bubbling up.
        """
        # For now, we emit the wrapped command if possible, or just the action.
        # But to be safe and ensure backward compatibility in the receiver (app.py),
        # we might want to emit the ICommand interface which provides .to_dict() etc.
        # But app.py likely expects the raw action object (Action struct from C++ or dict).

        # Pilot Implementation:
        # We wrap it, but ensure the receiver can handle it.
        # Since this is a partial rollout, let's assume the receiver checks for 'execute'.
        cmd = wrap_action(action)
        self.action_triggered.emit(cmd)

    def _open_popup(self, *args):
        from dm_toolkit.gui.widgets.zone_popup import ZonePopup
        # Pass civ_map if available
        civ_map = getattr(self, 'civ_map', None)
        popup = ZonePopup(self.title, self.last_card_data_list, self.card_db, self.civ_map, self.legal_actions, parent=self.window())

        # Track active popup to update it if game state changes while it's open
        self.active_popup = popup

        # Connect signals from popup's inner widget to our signals
        # So if user clicks a card in popup, it behaves as if they clicked it in the zone
        popup.zone_widget.card_clicked.connect(self.card_clicked.emit)
        popup.zone_widget.card_double_clicked.connect(self.card_double_clicked.emit)
        popup.zone_widget.action_triggered.connect(self.action_triggered.emit)
        popup.zone_widget.card_hovered.connect(self.card_hovered.emit)

        popup.exec()

        self.active_popup = None
