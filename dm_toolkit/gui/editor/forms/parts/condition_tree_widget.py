# -*- coding: utf-8 -*-
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QSplitter, QMenu, QMessageBox, QGroupBox, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.gui.editor.widgets.common import EditorWidgetMixin

class ConditionTreeWidget(QWidget, EditorWidgetMixin):
    """
    A widget for editing hierarchical conditions (AND/OR trees).
    Implements Proposal 1 (Hierarchical Logic Tree) + Proposal 4 (Context via ConditionEditorWidget).
    """
    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None # Context item for variable linking
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter for Tree vs Detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Side: Tree View ---
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_add_and = QPushButton(tr("AND Group"))
        self.btn_add_or = QPushButton(tr("OR Group"))
        self.btn_add_not = QPushButton(tr("NOT Group"))
        self.btn_add_leaf = QPushButton(tr("Condition"))
        self.btn_remove = QPushButton(tr("Remove"))

        toolbar.addWidget(self.btn_add_and)
        toolbar.addWidget(self.btn_add_or)
        toolbar.addWidget(self.btn_add_not)
        toolbar.addWidget(self.btn_add_leaf)
        toolbar.addWidget(self.btn_remove)
        toolbar.addStretch()

        tree_layout.addLayout(toolbar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel(tr("Logic Structure"))
        self.tree.currentItemChanged.connect(self.on_tree_selection_changed)
        tree_layout.addWidget(self.tree)

        tree_container.setLayout(tree_layout)
        splitter.addWidget(tree_container)

        # --- Right Side: Detail Editor ---
        self.detail_container = QGroupBox(tr("Condition Details"))
        detail_layout = QVBoxLayout(self.detail_container)

        self.leaf_editor = ConditionEditorWidget(title=tr("Condition Config"))
        self.leaf_editor.dataChanged.connect(self.on_leaf_data_changed)
        detail_layout.addWidget(self.leaf_editor)

        # Info label for groups
        self.group_info_label = QLabel(tr("Select a Group to see details."))
        self.group_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_layout.addWidget(self.group_info_label)

        splitter.addWidget(self.detail_container)

        # Set initial visibility
        self.leaf_editor.setVisible(False)
        self.group_info_label.setVisible(True)

        # Connect buttons
        self.btn_add_and.clicked.connect(lambda: self.add_item("AND"))
        self.btn_add_or.clicked.connect(lambda: self.add_item("OR"))
        self.btn_add_not.clicked.connect(lambda: self.add_item("NOT"))
        self.btn_add_leaf.clicked.connect(lambda: self.add_item("LEAF"))
        self.btn_remove.clicked.connect(self.remove_item)

        # Context Menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

    def set_current_item(self, item):
        """Sets the context item for variable linking."""
        self.current_item = item
        self.leaf_editor.set_current_item(item)

    def get_value(self):
        """Returns the JSON structure of the tree."""
        root = self.tree.invisibleRootItem()
        if root.childCount() == 0:
            return {}

        # If multiple roots, wrap in implicit AND or take first?
        # Usually logic tree has one root. If multiple, assume AND.
        if root.childCount() == 1:
            return self._item_to_json(root.child(0))
        else:
            children = []
            for i in range(root.childCount()):
                children.append(self._item_to_json(root.child(i)))
            return {"type": "AND", "children": children}

    def set_value(self, data):
        """Populates the tree from JSON structure."""
        self.tree.clear()
        if not data:
            return

        # If simple condition (no children/type is leaf-like), wrap or add
        # Check if it's a group
        ctype = data.get("type", "NONE")
        if ctype in ["AND", "OR", "NOT"]:
            self._json_to_item(data, self.tree.invisibleRootItem())
        else:
            # It's a leaf
            self._json_to_item(data, self.tree.invisibleRootItem())

        self.tree.expandAll()

    def _item_to_json(self, item):
        ctype = item.data(0, Qt.ItemDataRole.UserRole)
        data = item.data(0, Qt.ItemDataRole.UserRole + 1) or {}

        if ctype in ["AND", "OR", "NOT"]:
            children = []
            for i in range(item.childCount()):
                children.append(self._item_to_json(item.child(i)))
            return {"type": ctype, "children": children}
        else:
            # Leaf
            # data comes from leaf_editor update, but stored in item
            # Ensure type is set
            data['type'] = ctype
            return data

    def _json_to_item(self, data, parent_item):
        ctype = data.get("type", "NONE")

        item = QTreeWidgetItem(parent_item)
        item.setText(0, ctype)

        if ctype in ["AND", "OR", "NOT"]:
            item.setData(0, Qt.ItemDataRole.UserRole, ctype)
            # Process children
            children = data.get("children", [])
            for child in children:
                self._json_to_item(child, item)

            # NOT only has one child usually, but we handle list
        else:
            # Leaf
            item.setData(0, Qt.ItemDataRole.UserRole, ctype) # Store specific type or just 'LEAF'?
            # Actually, for leaf, the type in text should be the condition type (e.g. MANA_ARMED)
            # But user role tracks if it's a group or not?
            # Let's store the full data in UserRole + 1
            item.setData(0, Qt.ItemDataRole.UserRole + 1, data)
            item.setText(0, f"{ctype}")

        return item

    def on_tree_selection_changed(self, current, previous):
        if not current:
            self.leaf_editor.setVisible(False)
            self.group_info_label.setVisible(True)
            self.group_info_label.setText(tr("No selection"))
            return

        ctype = current.data(0, Qt.ItemDataRole.UserRole)

        if ctype in ["AND", "OR", "NOT"]:
            self.leaf_editor.setVisible(False)
            self.group_info_label.setVisible(True)
            self.group_info_label.setText(f"{tr('Group')}: {ctype}")
        else:
            # Leaf
            self.group_info_label.setVisible(False)
            self.leaf_editor.setVisible(True)

            data = current.data(0, Qt.ItemDataRole.UserRole + 1)
            self.leaf_editor.blockSignals(True)
            self.leaf_editor.set_data(data if data else {"type": ctype}) # Fallback
            self.leaf_editor.blockSignals(False)

    def on_leaf_data_changed(self):
        item = self.tree.currentItem()
        if not item: return

        data = self.leaf_editor.get_data()
        item.setData(0, Qt.ItemDataRole.UserRole + 1, data)

        # Update text
        ctype = data.get("type", "NONE")
        item.setData(0, Qt.ItemDataRole.UserRole, ctype) # Update type
        item.setText(0, ctype)

        self.dataChanged.emit()

    def add_item(self, item_type):
        parent = self.tree.currentItem()
        if not parent:
            parent = self.tree.invisibleRootItem()
        else:
            # If parent is leaf, add to its parent
            p_type = parent.data(0, Qt.ItemDataRole.UserRole)
            if p_type not in ["AND", "OR", "NOT"]:
                parent = parent.parent() or self.tree.invisibleRootItem()

        if item_type == "LEAF":
            data = {"type": "NONE"}
            self._json_to_item(data, parent)
        else:
            data = {"type": item_type, "children": []}
            self._json_to_item(data, parent)

        parent.setExpanded(True)
        self.dataChanged.emit()

    def remove_item(self):
        item = self.tree.currentItem()
        if not item: return

        parent = item.parent() or self.tree.invisibleRootItem()
        parent.removeChild(item)
        self.dataChanged.emit()

    def show_context_menu(self, position):
        menu = QMenu()
        menu.addAction(tr("Add AND"), lambda: self.add_item("AND"))
        menu.addAction(tr("Add OR"), lambda: self.add_item("OR"))
        menu.addAction(tr("Add Condition"), lambda: self.add_item("LEAF"))
        menu.addSeparator()
        menu.addAction(tr("Remove"), self.remove_item)
        menu.exec(self.tree.viewport().mapToGlobal(position))
