# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPainter
from PyQt6.QtCore import Qt, QRectF

class MCTSGraphView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
    def update_from_data(self, tree_data):
        self._scene.clear()
        if not tree_data:
            return

        # Simple tree layout
        # Calculate positions
        # We only draw top 2 levels
        
        root_node = tree_data
        children = root_node.get("children", [])
        
        # Root position
        root_x = 0
        root_y = 0
        
        self.draw_node(root_x, root_y, root_node, is_root=True)
        
        if not children:
            return

        # Layout children
        # Spread them horizontally
        width_per_child = 150
        total_width = len(children) * width_per_child
        start_x = -(total_width / 2) + (width_per_child / 2)
        
        for i, child in enumerate(children):
            child_x = start_x + i * width_per_child
            child_y = 150
            
            self.draw_edge(root_x, root_y, child_x, child_y)
            self.draw_node(child_x, child_y, child)
            
            # Grandchildren (optional, maybe just small dots?)
            grandchildren = child.get("children", [])
            if grandchildren:
                gc_width = 50
                gc_total_width = len(grandchildren) * gc_width
                gc_start_x = child_x - (gc_total_width / 2) + (gc_width / 2)
                
                for j, gc in enumerate(grandchildren):
                    gc_x = gc_start_x + j * gc_width
                    gc_y = 300
                    self.draw_edge(child_x, child_y, gc_x, gc_y)
                    self.draw_node(gc_x, gc_y, gc, small=True)

    def draw_node(self, x, y, data, is_root=False, small=False):
        radius = 30 if not small else 15
        rect = QRectF(x - radius, y - radius, radius * 2, radius * 2)
        
        # Color based on value
        value = data.get("value", 0.0)
        visits = data.get("visits", 0)
        avg_val = value / visits if visits > 0 else 0.0
        
        # Map [-1, 1] to [Red, Blue]
        # 0 -> White
        r, g, b = 255, 255, 255
        if avg_val > 0:
            # Blueish
            r = int(255 * (1 - avg_val))
            g = int(255 * (1 - avg_val))
            b = 255
        else:
            # Redish
            r = 255
            g = int(255 * (1 + avg_val))
            b = int(255 * (1 + avg_val))
            
        color = QColor(r, g, b)
        brush = QBrush(color)
        pen = QPen(Qt.GlobalColor.black)
        
        item = self._scene.addEllipse(rect, pen, brush)
        if item is not None:
            item.setToolTip(f"Action: {data.get('name')}\nVisits: {visits}\nValue: {avg_val:.2f}")
        
        if not small:
            text = data.get("name", "")
            # Truncate text
            if len(text) > 10: text = text[:8] + "..."
            
            text_item = QGraphicsTextItem(text)
            text_item.setFont(QFont("Arial", 8))
            text_rect = text_item.boundingRect()
            text_item.setPos(x - text_rect.width() / 2, y - text_rect.height() / 2)
            self._scene.addItem(text_item)
            
            # Visits count below
            visit_text = QGraphicsTextItem(str(visits))
            visit_text.setFont(QFont("Arial", 7))
            v_rect = visit_text.boundingRect()
            visit_text.setPos(x - v_rect.width() / 2, y + radius + 2)
            self._scene.addItem(visit_text)

    def draw_edge(self, x1, y1, x2, y2):
        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(QPen(Qt.GlobalColor.gray))
        line.setZValue(-1) # Behind nodes
        self._scene.addItem(line)

from PyQt6.QtGui import QPainter
