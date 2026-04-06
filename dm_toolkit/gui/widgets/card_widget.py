# -*- coding: utf-8 -*-
try:
    from PyQt6.QtWidgets import (
        QFrame, QVBoxLayout, QLabel, QHBoxLayout, QMenu
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QRect
    from PyQt6.QtGui import QAction, QCursor, QPainter, QColor, QPen
except Exception:
    # Headless / PyQt not installed: provide lightweight shims so tests can import GUI modules
    class _DummySignal:
        def __init__(self, *args, **kwargs):
            pass
        def emit(self, *a, **k):
            return None

    class _DummyWidget:
        def __init__(self, *args, **kwargs):
            pass
        def setFixedSize(self, *a, **k):
            pass
        def setFrameStyle(self, *a, **k):
            pass
        def setLineWidth(self, *a, **k):
            pass
        def setCursor(self, *a, **k):
            pass
        def setAccessibleName(self, *a, **k):
            pass
        def setAccessibleDescription(self, *a, **k):
            pass
        def setToolTip(self, *a, **k):
            pass
        def setStyleSheet(self, *a, **k):
            pass
        def setVisible(self, *a, **k):
            pass

    class _DummyLayout:
        def __init__(self, *a, **k):
            pass
        def setContentsMargins(self, *a, **k):
            pass
        def setSpacing(self, *a, **k):
            pass
        def addWidget(self, *a, **k):
            pass
        def addLayout(self, *a, **k):
            pass
        def addStretch(self, *a, **k):
            pass

    class _DummyLabel:
        def __init__(self, text=""):
            self._text = text
        def setFixedSize(self, *a, **k):
            pass
        def setAlignment(self, *a, **k):
            pass
        def setWordWrap(self, *a, **k):
            pass
        def setFont(self, *a, **k):
            pass
        def font(self):
            class F:
                def setBold(self, *a, **k): pass
                def setPointSize(self, *a, **k): pass
            return F()

    class _DummyMenu:
        def __init__(self, *a, **k): pass
        def addAction(self, *a, **k): pass
        def exec(self, *a, **k): pass

    class _DummyAction:
        def __init__(self, *a, **k): pass
        def triggered(self):
            return self
        def connect(self, *a, **k): pass

    class _DummyQt:
        class CursorShape:
            PointingHandCursor = None
        class MouseButton:
            LeftButton = 1
        class AlignmentFlag:
            AlignCenter = 0
            AlignTop = 0
            AlignLeft = 0
            AlignBottom = 0
            AlignRight = 0

    QFrame = _DummyWidget
    QVBoxLayout = _DummyLayout
    QLabel = _DummyLabel
    QHBoxLayout = _DummyLayout
    QMenu = _DummyMenu
    Qt = _DummyQt
    pyqtSignal = _DummySignal
    QAction = _DummyAction
    QCursor = None

from dm_toolkit.gui.styles.civ_colors import CIV_COLORS_FOREGROUND, CIV_COLORS_BACKGROUND
from dm_toolkit.gui.i18n import tr

# 再発防止: preview_pane.py からの import は循環依存リスクがあるため使用しない。
# ManaCostLabel と同等のパイチャートウィジェットをインラインで定義する。
class _CivCostOrb(QLabel):
    """コストサークルをパイチャートで文明色に塗り分けるラベル（多色対応）。"""
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._civs: list = []
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "font-weight: bold; font-size: 10px; color: white; "
            "background-color: transparent; padding: 0px;"
        )

    def set_civs(self, civs: list) -> None:
        self._civs = list(civs)
        self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = self.rect()
            side = min(r.width(), r.height())
            m = 1
            draw_rect = QRect(
                (r.width() - side) // 2 + m,
                (r.height() - side) // 2 + m,
                side - 2 * m,
                side - 2 * m,
            )
            if draw_rect.width() <= 0:
                painter.end()
                return
            civs = self._civs
            if not civs:
                painter.setBrush(QColor("#A9A9A9"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(draw_rect)
            elif len(civs) == 1:
                painter.setBrush(QColor(CIV_COLORS_FOREGROUND.get(civs[0], "#A9A9A9")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(draw_rect)
            else:
                n = len(civs)
                total = 360 * 16
                base = 90 * 16
                for i, civ in enumerate(civs):
                    painter.setBrush(QColor(CIV_COLORS_FOREGROUND.get(civ, "#A9A9A9")))
                    painter.setPen(Qt.PenStyle.NoPen)
                    a_start = base + (total * i) // n
                    a_end = base + (total * (i + 1)) // n
                    painter.drawPie(draw_rect, a_start, a_end - a_start)
            pen = QPen(Qt.GlobalColor.black)
            pen.setWidth(1)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawEllipse(draw_rect)
            font = self.font()
            font.setBold(True)
            font.setPixelSize(max(7, int(draw_rect.width() * 0.55)))
            painter.setFont(font)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(draw_rect, Qt.AlignmentFlag.AlignCenter, self.text())
            painter.end()
        except Exception:
            super().paintEvent(event)

class CardWidget(QFrame):
    clicked = pyqtSignal(int)  # Emits instance_id
    hovered = pyqtSignal(int)  # Emits card_id
    command_triggered = pyqtSignal(object)  # CommandDef を emit（旧: action_triggered）
    # 再発防止: シグナル名を command_triggered に統一。action_triggered は削除済み。
    double_clicked = pyqtSignal(int)  # Emits instance_id for quick play
    ICON_WIDTH = 100
    ICON_HEIGHT = 124

    def __init__(self, card_id, card_name, cost, power, civ, tapped=False,
                 instance_id=-1, parent=None, is_face_down=False, legal_commands=None):
        """
        civ: Can be a single string (e.g. "FIRE")
        or a list of strings (e.g. ["FIRE", "NATURE"]).
        legal_commands: List of CommandDef objects available for this card.
        """
        super().__init__(parent)
        self.card_id = card_id
        self.card_name = card_name
        self.cost = cost
        self.power = power
        self.legal_commands = legal_commands if legal_commands else []

        # Normalize civ to list
        if isinstance(civ, list):
            self.civs = civ
        else:
            self.civs = [civ] if civ else []

        self.tapped = tapped
        self.selected = False       # 選択状態（対象選択時の確定選択）
        self.highlight_legal = False   # 方針A: 操作可能カード（緑枠）
        self.highlight_target = False  # 方針A: 有効対象カード（黄枠）
        self.instance_id = instance_id
        self.is_face_down = is_face_down

        self.setFixedSize(self.ICON_WIDTH, self.ICON_HEIGHT)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)

        # UX Improvement: Cursor Feedback
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # UX Improvement: Accessibility (localized)
        self.setAccessibleName(f"{tr('CARD')}: {self.card_name}")
        civ_str = "/".join(self.civs)
        self.setAccessibleDescription(
            f"{tr('Cost')} {self.cost}, {tr('Power')} {self.power}, {tr('Civ')} {civ_str}"
        )

        self.setToolTip(
            f"{tr('Name')}: {self.card_name}\n"
            f"{tr('Cost')}: {self.cost}\n"
            f"{tr('Power')}: {self.power}\n"
            f"{tr('Civ')}: {civ_str}"
        )

        self.init_ui()
        # 暗転オーバーレイ: タップ時に黒半透明フレームを重ねて明度を下げる
        # 再発防止: QGraphicsOpacityEffect は薄くする（透過）効果のため使用しない
        self._tap_overlay = QFrame(self)
        self._tap_overlay.setFixedSize(self.ICON_WIDTH, self.ICON_HEIGHT)
        self._tap_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 130); border-radius: 5px;"
        )
        self._tap_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tap_overlay.setVisible(False)
        self.update_style()
        self._apply_tap_effect()

    def _apply_tap_effect(self):
        """タップ済みカードに暗転オーバーレイを表示して明度を下げる。アンタップ時は非表示。"""
        try:
            self._tap_overlay.setVisible(self.tapped)
            if self.tapped:
                self._tap_overlay.raise_()
        except Exception:
            pass

    def update_legal_commands(self, commands):
        self.legal_commands = commands

    # 後方互換エイリアス（段階的廃止）
    def update_legal_actions(self, actions):
        self.update_legal_commands(actions)

    def enterEvent(self, event):
        self.hovered.emit(self.card_id)
        super().enterEvent(event)

    def mousePressEvent(self, event):
        """Handle left-click to emit clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.instance_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to quickly play the default action."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.instance_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu on right click."""
        if not self.legal_commands:
            return

        menu = QMenu(self)

        # Categorize commands
        # Simplified categorization logic
        added_types = set()

        for action in self.legal_commands:
            action_str = action.to_string() # Fallback description

            # Simple heuristic for display text (localized)
            label = action_str
            if "Play" in action_str:
                label = tr("Play Card")
            elif "Attack" in action_str:
                if "Player" in action_str:
                    label = tr("Attack Player")
                elif "Creature" in action_str:
                    label = tr("Attack Creature")
                else:
                    label = tr("Attack")
            elif "Mana" in action_str:
                label = tr("MANA_CHARGE")
            elif "Use Ability" in action_str:
                label = tr("Use Ability")

            # De-duplicate identical labels if multiple similar actions exist (e.g. attack different shields)
            # For simplicity, if we have multiple Attack Player (different shields), we might want to just show one "Attack Player"
            # and let the engine resolve/ask target, BUT the engine usually generates distinct actions.
            # For now, list them all but try to be descriptive.

                act = QAction(label, self)
                # Use a closure to capture the specific action
                safe_connect(act, 'triggered', lambda checked, a=action: self.command_triggered.emit(a))
            menu.addAction(act)

        menu.exec(event.globalPos())

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header (Cost Circle + Name)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(2)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Cost Circle — _CivCostOrb でパイチャート表示（プレビューと統一）
        # 再発防止: 多色カードで最初の文明色のみ使う問題を解消
        self.cost_label = _CivCostOrb(str(self.cost))
        self.cost_label.setFixedSize(24, 24)
        self.cost_label.set_civs(self.civs)

        # Hide cost label if face down
        if self.is_face_down:
            self.cost_label.setVisible(False)

        # Style will be set in update_style

        header_layout.addWidget(self.cost_label)

        # Name
        self.name_label = QLabel(self.card_name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        font = self.name_label.font()
        font.setBold(True)
        font.setPointSize(8)
        self.name_label.setFont(font)

        header_layout.addWidget(self.name_label)

        layout.addLayout(header_layout)

        layout.addStretch()

        # Footer (Power)
        if self.power > 0:
            self.power_label = QLabel(f"BP:{self.power}")
            self.power_label.setAlignment(
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
            )
            font = self.power_label.font()
            font.setPointSize(7)
            self.power_label.setFont(font)
            layout.addWidget(self.power_label)

    def get_civ_color(self, civ):
        return CIV_COLORS_FOREGROUND.get(civ, "#A9A9A9")

    def get_bg_civ_color(self, civ):
        # Lighter colors for card background
        return CIV_COLORS_BACKGROUND.get(civ, "#FFFFFF")

    def _build_multicolor_gradient(self) -> str:
        """Build a qlineargradient string using all civilizations on the card."""
        if not self.civs:
            return "background-color: #FFFFFF;"
        if len(self.civs) == 1:
            return f"background-color: {self.get_bg_civ_color(self.civs[0])};"

        # 再発防止: 多色カードで最初の2文明だけを使うと表示が不整合になるため、
        # 全文明を stop に展開してプレビューと同様の見た目に寄せる。
        n = len(self.civs)
        stops = []
        for i, civ in enumerate(self.civs):
            color = self.get_bg_civ_color(civ)
            pos = i / (n - 1)
            stops.append(f"stop:{pos:.3f} {color}")
        stop_str = ", ".join(stops)
        return (
            "background: qlineargradient(spread:pad, "
            f"x1:0, y1:0, x2:1, y2:1, {stop_str});"
        )

    def update_style(self):
        # 1. Cost Circle 更新（_CivCostOrb は set_civs で再描画）
        if hasattr(self.cost_label, 'set_civs'):
            self.cost_label.set_civs(self.civs)
        else:
            c = CIV_COLORS_FOREGROUND.get(self.civs[0], "#A9A9A9") if self.civs else "#A9A9A9"
            self.cost_label.setStyleSheet(
                f"font-weight: bold; font-size: 10px; color: white; "
                f"border: 1px solid black; border-radius: 12px; background-color: {c};"
            )


        # 2. Update Card Background Style
        border_color = '#555'
        border_width = '2px'

        if self.selected:
            border_color = '#FF3333'  # 確定選択: 赤枠
            border_width = '4px'
        elif self.highlight_target:
            border_color = '#FFD700'  # 有効対象: 黄枠（方針A）
            border_width = '3px'
        elif self.highlight_legal:
            border_color = '#00CC44'  # 操作可能: 明るい緑枠（方針A）
            border_width = '3px'

        bg_style = self._build_multicolor_gradient()

        # UX Improvement: Add hover style
        self.setStyleSheet(f"""
            CardWidget {{
                {bg_style}
                border: {border_width} solid {border_color};
                border-radius: 5px;
            }}
            CardWidget:hover {{
                border: {border_width} solid {'#0078d7' if not self.selected else '#FF3333'};
            }}
        """)

    def set_tapped(self, tapped):
        self.tapped = tapped
        self.update_style()
        self._apply_tap_effect()

    def set_selected(self, selected):
        self.selected = selected
        self.update_style()

    def set_highlight_legal(self, val: bool):
        """操作可能カードとして緑枠ハイライトする（方針A）。"""
        self.highlight_legal = val
        self.update_style()

    def set_highlight_target(self, val: bool):
        """有効対象カードとして黄枠ハイライトする（方針A）。"""
        self.highlight_target = val
        self.update_style()

    def clear_highlights(self):
        """全ハイライトをリセットする。"""
        self.highlight_legal = False
        self.highlight_target = False
        self.update_style()

    def mousePressEvent(self, a0):
        if a0 and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.instance_id)
        super().mousePressEvent(a0)
