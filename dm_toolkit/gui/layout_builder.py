from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QDockWidget, QToolBar
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.widgets.scenario_tools import ScenarioToolsDock
from dm_toolkit.gui.widgets.mcts_view import MCTSView
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel
from dm_toolkit.gui.widgets.stack_view import StackViewWidget
from dm_toolkit.gui.widgets.loop_recorder import LoopRecorderWidget
from dm_toolkit.gui.widgets.card_effect_debugger import CardEffectDebugger
from dm_toolkit.gui.widgets.control_panel import ControlPanel
from dm_toolkit.gui.widgets.game_board import GameBoard

if TYPE_CHECKING:
    from dm_toolkit.gui.app import GameWindow

class LayoutBuilder:
    def __init__(self, window: 'GameWindow'):
        self.window = window

    def build(self):
        self.init_toolbar()
        self.init_ui()

    def init_toolbar(self):
        window = self.window
        window.toolbar = QToolBar(tr("Main Toolbar"), window)
        window.toolbar.setObjectName("MainToolbar")
        window.addToolBar(window.toolbar)

        deck_act = QAction(tr("Deck Builder"), window)
        deck_act.triggered.connect(window.open_deck_builder)
        window.toolbar.addAction(deck_act)

        card_act = QAction(tr("Card Editor"), window)
        card_act.triggered.connect(window.open_card_editor)
        window.toolbar.addAction(card_act)

        window.scen_act = QAction(tr("Scenario Mode"), window)
        window.scen_act.setCheckable(True)
        window.scen_act.triggered.connect(window.toggle_scenario_mode)
        window.toolbar.addAction(window.scen_act)

        sim_act = QAction(tr("Batch Simulation"), window)
        sim_act.triggered.connect(window.open_simulation_dialog)
        window.toolbar.addAction(sim_act)

        ai_act = QAction(tr("AI Analysis"), window)
        # Assuming mcts_dock is created in init_ui, we use lambda to access it at runtime
        ai_act.triggered.connect(lambda: window.mcts_dock.setVisible(not window.mcts_dock.isVisible()))
        window.toolbar.addAction(ai_act)

        loop_act = QAction(tr("Loop Recorder"), window)
        loop_act.triggered.connect(lambda: window.loop_dock.setVisible(not window.loop_dock.isVisible()))
        window.toolbar.addAction(loop_act)

        debug_act = QAction(tr("Effect Debugger"), window)
        debug_act.triggered.connect(lambda: window.debugger_dock.setVisible(not window.debugger_dock.isVisible()))
        window.toolbar.addAction(debug_act)

        log_act = QAction(tr("Logs"), window)
        # window.log_dock is created in init_ui
        log_act.triggered.connect(lambda: window.log_dock.setVisible(not window.log_dock.isVisible()))
        window.toolbar.addAction(log_act)

    def init_ui(self):
        window = self.window

        # AI Tools Dock (Left)
        window.ai_tools_dock = QDockWidget(tr("AI & Tools"), window)
        window.ai_tools_dock.setObjectName("AIToolsDock")
        window.ai_tools_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        window.ai_tools_panel = QWidget()
        window.ai_tools_panel.setMinimumWidth(300)
        window.ai_tools_layout = QVBoxLayout(window.ai_tools_panel)
        window.ai_tools_dock.setWidget(window.ai_tools_panel)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, window.ai_tools_dock)

        # Game Status Dock (Right)
        window.status_dock = QDockWidget(tr("Game Status & Operations"), window)
        window.status_dock.setObjectName("StatusDock")
        window.status_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        window.status_panel = QWidget()
        window.status_panel.setMinimumWidth(300)
        window.status_layout_main = QVBoxLayout(window.status_panel)
        window.status_dock.setWidget(window.status_panel)

        window.top_section_group = QGroupBox(tr("Game Status"))
        top_layout = QVBoxLayout()

        status_layout = QHBoxLayout()
        window.turn_label = QLabel(tr("Turn: {turn}").format(turn=1))
        window.turn_label.setStyleSheet("font-weight: bold;")
        window.phase_label = QLabel(tr("Phase: {phase}").format(phase=tr("Start Phase")))
        window.active_label = QLabel(tr("Active: P{player_id}").format(player_id=0))
        status_layout.addWidget(window.turn_label)
        status_layout.addWidget(window.phase_label)
        status_layout.addWidget(window.active_label)
        top_layout.addLayout(status_layout)

        window.card_detail_panel = CardDetailPanel()
        top_layout.addWidget(window.card_detail_panel)

        window.top_section_group.setLayout(top_layout)
        window.status_layout_main.addWidget(window.top_section_group)

        # Control Panel
        window.control_panel = ControlPanel()
        window.control_panel.start_simulation_clicked.connect(window.toggle_simulation)
        window.control_panel.step_clicked.connect(window.session.step_phase)
        window.control_panel.pass_clicked.connect(window.pass_turn)
        window.control_panel.confirm_clicked.connect(window.confirm_selection)
        window.control_panel.reset_clicked.connect(window.reset_game)

        # Tool connections
        window.control_panel.deck_builder_clicked.connect(window.open_deck_builder)
        window.control_panel.card_editor_clicked.connect(window.open_card_editor)
        window.control_panel.batch_sim_clicked.connect(window.open_simulation_dialog)
        window.control_panel.load_deck_p0_clicked.connect(window.load_deck_p0)
        window.control_panel.load_deck_p1_clicked.connect(window.load_deck_p1)
        window.control_panel.god_view_toggled.connect(window.update_ui)
        window.control_panel.help_clicked.connect(window.show_help)

        # Mode update connections
        window.control_panel.p0_human_radio.toggled.connect(lambda c: window.session.set_player_mode(0, 'Human' if c else 'AI'))
        window.control_panel.p1_human_radio.toggled.connect(lambda c: window.session.set_player_mode(1, 'Human' if c else 'AI'))

        window.ai_tools_layout.addWidget(window.control_panel)
        window.ai_tools_layout.addStretch()

        window.status_layout_main.addStretch()

        # Board Panel
        window.game_board = GameBoard()
        window.game_board.action_triggered.connect(window.session.execute_action)
        window.game_board.card_clicked.connect(window.on_card_clicked)
        window.game_board.card_double_clicked.connect(window.on_card_double_clicked)
        window.game_board.card_hovered.connect(window.on_card_hovered)

        window.setCentralWidget(window.game_board)

        # Docks
        window.stack_dock = QDockWidget(tr("Pending Effects"), window)
        window.stack_dock.setObjectName("StackDock")
        window.stack_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        window.stack_view = StackViewWidget()
        window.stack_view.effect_resolved.connect(window.on_resolve_effect_from_stack)
        window.stack_dock.setWidget(window.stack_view)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.stack_dock)

        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.status_dock)
        window.splitDockWidget(window.stack_dock, window.status_dock, Qt.Orientation.Vertical)

        window.log_dock = QDockWidget(tr("Logs"), window)
        window.log_dock.setObjectName("LogDock")
        window.log_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        # log_viewer is already created in GameWindow.__init__
        window.log_dock.setWidget(window.log_viewer)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.log_dock)
        window.log_dock.hide()

        window.mcts_dock = QDockWidget(tr("MCTS Analysis"), window)
        window.mcts_dock.setObjectName("MCTSDock")
        window.mcts_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        window.mcts_view = MCTSView()
        window.mcts_dock.setWidget(window.mcts_view)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.mcts_dock)
        window.mcts_dock.hide()

        window.loop_dock = QDockWidget(tr("Loop Recorder"), window)
        window.loop_dock.setObjectName("LoopDock")
        window.loop_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        window.loop_recorder = LoopRecorderWidget(lambda: window.gs)
        window.loop_dock.setWidget(window.loop_recorder)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, window.loop_dock)
        window.loop_dock.hide()

        window.debugger_dock = QDockWidget(tr("Card Effect Debugger"), window)
        window.debugger_dock.setObjectName("DebuggerDock")
        window.debugger_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        window.effect_debugger = CardEffectDebugger()
        window.debugger_dock.setWidget(window.effect_debugger)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.debugger_dock)
        window.debugger_dock.hide()

        window.scenario_tools = ScenarioToolsDock(window, window.gs, window.card_db)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, window.scenario_tools)
        window.scenario_tools.hide()
