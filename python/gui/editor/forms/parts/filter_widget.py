from PyQt6.QtWidgets import QWidget, QGroupBox, QGridLayout, QLabel, QCheckBox, QComboBox, QSpinBox
from PyQt6.QtCore import pyqtSignal
from gui.localization import tr

class FilterEditorWidget(QWidget):
    """
    Reusable widget for editing FilterDef properties.
    Handles Zone selection, Count mode (All/Any vs Fixed), and related fields.
    """

    # Signal emitted when any filter property changes
    filterChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # We use a GridLayout directly on this widget, or wrap in GroupBox?
        # Since the parent usually wraps this in a GroupBox or layout, let's keep it simple.
        # But ActionEditForm uses a GroupBox "Filter". Let's inherit that structure or render inside it.
        # Let's make this widget *contain* the layout that goes inside the GroupBox.

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Help Label
        help_label = QLabel(tr("Filter Help"))
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(help_label, 0, 0, 1, 2)

        # Zones
        self.zone_checks = {}
        zones = ["BATTLE_ZONE", "MANA_ZONE", "HAND", "GRAVEYARD", "SHIELD_ZONE", "DECK"]
        for i, z in enumerate(zones):
            cb = QCheckBox(tr(z))
            cb.setToolTip(tr(f"Include {z} in target selection"))
            self.zone_checks[z] = cb
            layout.addWidget(cb, (i//2) + 1, i%2)
            cb.stateChanged.connect(self.filterChanged.emit)

        # Selection Mode
        self.mode_label = QLabel(tr("Selection Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("All/Any"), 0)
        self.mode_combo.addItem(tr("Fixed Number"), 1)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 99)
        self.count_spin.setToolTip(tr("Number of cards to select/count."))
        self.count_spin.setVisible(False) # Default hidden

        layout.addWidget(self.mode_label, 4, 0)
        layout.addWidget(self.mode_combo, 4, 1)
        layout.addWidget(self.count_spin, 5, 1)

        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.count_spin.valueChanged.connect(self.filterChanged.emit)

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_fixed = (mode == 1)
        self.count_spin.setVisible(is_fixed)
        self.filterChanged.emit()

    def set_data(self, filt_data):
        """
        Populate UI from dictionary (FilterDef).
        """
        self.blockSignals(True)

        zones = filt_data.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        count_val = filt_data.get('count', 0)
        if count_val > 0:
            self.mode_combo.setCurrentIndex(1) # Fixed
            self.count_spin.setValue(count_val)
            self.count_spin.setVisible(True)
        else:
            self.mode_combo.setCurrentIndex(0) # All/Any
            self.count_spin.setValue(1)
            self.count_spin.setVisible(False)

        self.blockSignals(False)

    def get_data(self):
        """
        Return dictionary (FilterDef) from UI.
        """
        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]

        filt = {}
        if zones: filt['zones'] = zones

        mode = self.mode_combo.currentData()
        if mode == 1:
            count = self.count_spin.value()
            if count > 0: filt['count'] = count

        return filt
