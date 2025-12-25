## 2024-05-23 - Simulation Dialog Accessibility
**Learning:** PyQt6 widgets like QSpinBox and QComboBox should have associated labels using `setBuddy` or explicit layouts to ensure screen reader accessibility, even if visually labeled.
**Action:** When adding labels to form controls, use `label.setBuddy(widget)` or ensure the layout clearly associates them.

## 2024-05-23 - Dialog Focus & Default Actions
**Learning:** Users expect the 'OK' button to be triggered by Enter (via `setDefault(True)`) and for focus to land on the primary interaction element (the list) immediately upon opening a selection dialog.
**Action:** Always set `setDefault(True)` on the primary action button and explicitly `setFocus()` on the main input widget in `__init__`.
