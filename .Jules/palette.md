## 2024-05-23 - Simulation Dialog Accessibility
**Learning:** PyQt6 widgets like QSpinBox and QComboBox should have associated labels using `setBuddy` or explicit layouts to ensure screen reader accessibility, even if visually labeled.
**Action:** When adding labels to form controls, use `label.setBuddy(widget)` or ensure the layout clearly associates them.
