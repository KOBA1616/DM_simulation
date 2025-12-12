from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.editor.forms.card_form import CardEditForm
from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm
from dm_toolkit.gui.editor.forms.action_form import ActionEditForm
from dm_toolkit.gui.editor.forms.spell_side_form import SpellSideForm
from dm_toolkit.gui.localization import tr

class PropertyInspector(QWidget):
    # Forward signal from forms
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.label = QLabel(tr("Property Inspector"))
        layout.addWidget(self.label)

        self.stack = QStackedWidget()

        self.empty_page = QLabel(tr("Select an item to edit"))
        self.stack.addWidget(self.empty_page)

        self.card_form = CardEditForm()
        self.stack.addWidget(self.card_form)
        self.card_form.structure_update_requested.connect(self.structure_update_requested.emit)

        self.effect_form = EffectEditForm()
        self.stack.addWidget(self.effect_form)

        self.action_form = ActionEditForm()
        self.stack.addWidget(self.action_form)

        self.spell_side_form = SpellSideForm()
        self.stack.addWidget(self.spell_side_form)

        layout.addWidget(self.stack)

    def set_selection(self, index):
        if index is None or not index.isValid():
            self.stack.setCurrentWidget(self.empty_page)
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        item = index.model().itemFromIndex(index)

        if item_type == "CARD":
            self.card_form.set_data(item)
            self.stack.setCurrentWidget(self.card_form)
        elif item_type == "EFFECT":
            self.effect_form.set_data(item)
            self.stack.setCurrentWidget(self.effect_form)
        elif item_type == "ACTION":
            self.action_form.set_data(item)
            self.stack.setCurrentWidget(self.action_form)
        elif item_type == "SPELL_SIDE":
            self.spell_side_form.set_data(item)
            self.stack.setCurrentWidget(self.spell_side_form)
        else:
            self.stack.setCurrentWidget(self.empty_page)
