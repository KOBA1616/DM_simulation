# Palette's Journal - Critical UX/A11y Learnings

## 2024-05-22 - Standard Icons and Menus in PyQt6

**Learning:** `QStyle.StandardPixmap` provides immediate, native-looking icons for common actions (New, Save, Delete) without needing external assets. This is a huge win for rapid prototyping and ensuring native feel.
**Action:** Use `self.style().standardIcon(QStyle.StandardPixmap.SP_...)` for all standard actions before considering custom icons.

## 2024-05-22 - Splitter Usability

**Learning:** Default `QSplitter` handles are often too thin (invisible click targets) and lack hover states. Styling `QSplitter::handle` with a wider width and hover color drastically improves discoverability.
**Action:** Always add a basic stylesheet to splitters to ensure they are visible and usable.
