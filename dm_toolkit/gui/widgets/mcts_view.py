from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QProgressBar, QHeaderView, QTabWidget
from PyQt6.QtCore import Qt
from .mcts_graph_view import MCTSGraphView

class MCTSView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.label = QLabel("AI思老E�Eロセス (MCTS)")
        self.label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Tab 1: Tree Table
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["アクション", "訪問回数", "評価値 (Q)", "優先度 (P)"])
        header = self.tree.header()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setColumnWidth(0, 200)
        self.tabs.addTab(self.tree, "チE�Eブル表示")
        
        # Tab 2: Graph View
        self.graph_view = MCTSGraphView()
        self.tabs.addTab(self.graph_view, "グラフ表示")
        
        self.win_rate_bar = QProgressBar()
        self.win_rate_bar.setRange(0, 100)
        self.win_rate_bar.setFormat("勝率予測: %p%")
        layout.addWidget(self.win_rate_bar)

    def update_from_data(self, tree_data):
        self.tree.clear()
        self.graph_view.update_from_data(tree_data)
        
        if not tree_data:
            return

        # tree_data is a dict with keys: name, visits, value, children
        # children is a list of dicts
        
        root_visits = tree_data.get("visits", 0)
        root_value = tree_data.get("value", 0)
        
        if root_visits > 0:
            # Value is sum of rewards. Average value = value / visits.
            # Reward is [0, 1]. So avg is [0, 1].
            avg_val = root_value / root_visits
            win_rate = avg_val * 100
            self.win_rate_bar.setValue(int(win_rate))
        
        children = tree_data.get("children", [])
        
        for child in children:
            action_str = child.get("name", "Unknown")
            visits = child.get("visits", 0)
            value_sum = child.get("value", 0.0)
            avg_val = value_sum / visits if visits > 0 else 0.0
            
            item = QTreeWidgetItem([
                action_str,
                str(visits),
                f"{avg_val:.3f}",
                "-" # Prior not implemented in Python MCTS
            ])
            self.tree.addTopLevelItem(item)
            
            # Add grandchildren (optional, maybe too much detail)
            grandchildren = child.get("children", [])
            for gc in grandchildren:
                gc_action = gc.get("name", "Unknown")
                gc_visits = gc.get("visits", 0)
                gc_val = gc.get("value", 0.0)
                gc_avg = gc_val / gc_visits if gc_visits > 0 else 0.0
                
                gc_item = QTreeWidgetItem([
                    gc_action,
                    str(gc_visits),
                    f"{gc_avg:.3f}",
                    "-"
                ])
                item.addChild(gc_item)
