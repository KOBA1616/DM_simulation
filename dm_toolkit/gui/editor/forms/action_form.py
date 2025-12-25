"""
Legacy `ActionEditForm` implementation preserved for reference but deprecated.
The editor now uses `UnifiedActionForm` for both Action and Command editing.
This module provides a lightweight shim that delegates to `UnifiedActionForm`.
"""

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.localization import tr
import warnings


class ActionEditForm(UnifiedActionForm):
    def __init__(self, parent=None):
        warnings.warn("ActionEditForm is deprecated: using UnifiedActionForm instead", DeprecationWarning)
        super().__init__(parent)

    # Keep API surface compatible by inheriting UnifiedActionForm
    # Any legacy-specific behavior should be handled in UnifiedActionForm itself.
        self.source_zone_combo.blockSignals(block)
        self.dest_zone_combo.blockSignals(block)
        self.allow_duplicates_check.blockSignals(block)
        self.no_cost_check.blockSignals(block)
