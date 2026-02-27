# -*- coding: utf-8 -*-
"""
Simplified GameSession that delegates game progression to C++ engine.

Key principles:
1. C++ handles all automatic game progression via fast_forward()
2. Python only handles:
   - Human player input/output
   - UI updates
   - Initial setup
3. AI player logic should ideally be in C++ or called from C++
"""
from typing import Any, List, Optional, Callable, Dict
import random
import os
import sys

from dm_toolkit.dm_types import GameState, CardDB
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.gui.i18n import tr

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None


class GameSessionSimplified:
    """Simplified game session that leverages C++ engine for game progression."""
    
    DEFAULT_DECK: List[int] = []

    def __init__(self,
                 callback_update_ui: Optional[Callable[[], None]] = None,
                 callback_log: Optional[Callable[[str], None]] = None,
                 callback_input_request: Optional[Callable[[], None]] = None,
                 callback_action_executed: Optional[Callable[[Any], None]] = None,
                 select_number_mode: str = 'heuristic'):
        self.callback_update_ui = callback_update_ui or (lambda: None)
        self.callback_log = callback_log or (lambda m: None)
        self.callback_input_request = callback_input_request or (lambda: None)
        self.callback_action_executed = callback_action_executed or (lambda a: None)

        self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
        self.is_running = False
        self.is_processing = False
        self.card_db: CardDB = {}
        self.gs: Optional[GameState] = None
        
        # SELECT_NUMBER evaluation mode: 'heuristic', 'evaluator', 'mcts'
        # Can be overridden by DM_SELECT_NUMBER_MODE environment variable
        env_mode = os.getenv('DM_SELECT_NUMBER_MODE', '').lower()
        if env_mode in ['heuristic', 'evaluator', 'mcts']:
            self.select_number_mode = env_mode
        else:
            self.select_number_mode = select_number_mode
        
        self.evaluator = None
        self.mcts = None

    def initialize_game(self, card_db: CardDB, seed: int = 42) -> None:
        """Initialize game with C++ engine."""
        self.card_db = card_db
        if not dm_ai_module:
            self.callback_log("Error: dm_ai_module not available")
            return
        
        # Log selected mode
        self.callback_log(f"SELECT_NUMBER AI mode: {self.select_number_mode}")
        
        # Initialize evaluators if needed
        if self.select_number_mode == 'evaluator' and not self.evaluator:
            try:
                self.evaluator = dm_ai_module.HeuristicEvaluator(card_db)
                self.callback_log("HeuristicEvaluator initialized")
            except Exception as e:
                self.callback_log(f"Failed to init evaluator: {e}, fallback to heuristic")
                self.select_number_mode = 'heuristic'
        
        if self.select_number_mode == 'mcts' and not self.mcts:
            try:
                self.mcts = dm_ai_module.MCTS(card_db, 1.4, 0.3, 0.25, 1, 0.0)
                self.callback_log("MCTS initialized for SELECT_NUMBER evaluation")
            except Exception as e:
                self.callback_log(f"Failed to init MCTS: {e}, fallback to heuristic")
                self.select_number_mode = 'heuristic'

        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        # Set decks
        deck0 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        deck1 = list(self.DEFAULT_DECK) if self.DEFAULT_DECK else self._build_default_deck()
        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        # Start game via C++ PhaseManager
        try:
            dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
            self.callback_log("Game initialized via C++ PhaseManager.start_game")
        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_update_ui()

    def reset_game(self, p0_deck: Optional[List[int]] = None, p1_deck: Optional[List[int]] = None) -> None:
        """Reset game state."""
        if not dm_ai_module:
            return

        seed = random.randint(0, 100000)
        self.gs = dm_ai_module.GameState(seed)
        self.gs.setup_test_duel()

        deck0 = p0_deck if p0_deck else list(self.DEFAULT_DECK)
        deck1 = p1_deck if p1_deck else list(self.DEFAULT_DECK)
        
        if not deck0:
            deck0 = self._build_default_deck()
        if not deck1:
            deck1 = self._build_default_deck()

        self.gs.set_deck(0, deck0)
        self.gs.set_deck(1, deck1)

        try:
            dm_ai_module.PhaseManager.start_game(self.gs, self.card_db)
        except Exception as e:
            self.callback_log(f"start_game failed: {e}")

        self.callback_log(tr("Game Reset"))
        self.callback_update_ui()

    def step_phase(self):
        """Alias for step_game() for backward compatibility."""
        self.step_game()

    def step_game(self):
        """
        Main game loop - simplified to delegate to C++ engine.
        
        Flow:
        1. Check if game is over
        2. Get legal commands from C++ engine
        3. If human player: show options and wait for input
        4. If AI player: let C++ fast_forward handle it automatically
        """
        if self.is_processing or not self.gs:
            return

        self.is_processing = True
        try:
            # Check for user input wait state
            if self._check_and_handle_input_wait():
                return
            # Check game over
            if self.is_game_over():
                self.callback_log("Game Over")
                self.callback_update_ui()
                return

            active_pid = EngineCompat.get_active_player_id(self.gs)
            is_human = (self.player_modes.get(active_pid) == 'Human')

            # Get legal commands from C++ engine
            from dm_toolkit import commands
            cmds = commands.generate_legal_commands(self.gs, self.card_db)

            if not cmds:
                # No commands - let C++ fast_forward progress the game
                self._fast_forward()
                self.callback_update_ui()
                return

            if is_human:
                # Human player - wait for input
                # UI will call execute_action() when user selects an action
                self.callback_update_ui()
                return

            # AI player - use C++ to handle automatically
            # For now, execute first non-PASS action, then fast_forward
            best_cmd = self._select_ai_action(cmds)
            if best_cmd:
                self.execute_action(best_cmd)
                # After AI action, fast_forward to next decision point
                self._fast_forward()

            self.callback_update_ui()

        finally:
            self.is_processing = False

    def execute_action(self, raw_action: Any):
        """
        Execute an action and let C++ engine progress automatically.
        
        This method:
        1. Converts action to command dict
        2. Executes via C++ engine
        3. Calls fast_forward to progress to next decision point
        """
        if not self.gs:
            return

        try:
            # Convert to command dict
            cmd_dict = ensure_executable_command(raw_action)
        except Exception as e:
            self.callback_log(f"Command conversion error: {e}")
            return

        active_pid = EngineCompat.get_active_player_id(self.gs)

        try:
            # Execute command using appropriate C++ command class
            if (cmd_dict.get('type') == 'PLAY_FROM_ZONE' or cmd_dict.get('legacy_original_type') == 'DECLARE_PLAY') and \
                 dm_ai_module and hasattr(dm_ai_module, 'PlayCardCommand'):
                instance_id = int(cmd_dict.get('instance_id') or cmd_dict.get('source_instance_id', -1))
                cpp_cmd = dm_ai_module.PlayCardCommand(instance_id)
                self.gs.execute_command(cpp_cmd)
                self.callback_log(f"P{active_pid}: PLAY_CARD")
                
            elif cmd_dict.get('type') == 'PASS' and dm_ai_module and hasattr(dm_ai_module, 'PassCommand'):
                cpp_cmd = dm_ai_module.PassCommand()
                self.gs.execute_command(cpp_cmd)
                self.callback_log(f"P{active_pid}: PASS")
                
            else:
                # Unified fallback: handles MANA_CHARGE and all other types
                # via CommandSystem.execute_command which enforces game rules
                EngineCompat.ExecuteCommand(self.gs, cmd_dict, self.card_db)
                cmd_type = cmd_dict.get('type', 'UNKNOWN')
                self.callback_log(f"P{active_pid}: {cmd_type}")

            if self.callback_action_executed:
                self.callback_action_executed(cmd_dict)

        except Exception as e:
            self.callback_log(f"Execution error: {e}")

        # After executing action, fast_forward to next decision point
        self._fast_forward()
        self.callback_update_ui()

    def _fast_forward(self):
        """Call C++ fast_forward to progress game until next decision point."""
        if not dm_ai_module or not hasattr(dm_ai_module.PhaseManager, 'fast_forward'):
            return
        
        try:
            dm_ai_module.PhaseManager.fast_forward(self.gs, self.card_db)
        except Exception as e:
            self.callback_log(f"fast_forward error: {e}")

    def _select_ai_action(self, cmds: List[Any]) -> Any:
        """Select best action for AI player. Prefer non-PASS, non-NONE actions."""
        # Handle SELECT_NUMBER: choose optimal value based on game state
        for cmd in cmds:
            try:
                d = cmd.to_dict()
                cmd_type = d.get('type')
                
                # Auto-select for SELECT_NUMBER with heuristic evaluation
                if cmd_type == 'SELECT_NUMBER':
                    max_val = d.get('slot_index', 1)
                    best_choice = self._evaluate_select_number_options(max_val)
                    cmd.target_instance_id = best_choice
                    self.callback_log(f"AI [{self.select_number_mode}] selected {best_choice}/{max_val} for SELECT_NUMBER")
                    return cmd
            except Exception:
                pass
        
        # Normal action selection
        for cmd in cmds:
            try:
                d = cmd.to_dict()
                cmd_type = d.get('type')
                # Skip NONE type with legacy_warning (e.g., PAY_COST)
                if cmd_type == 'NONE' and d.get('legacy_warning'):
                    continue
                # Prefer non-PASS actions
                if cmd_type != 'PASS':
                    return cmd
            except Exception:
                pass
        
        # If only PASS available, return first command
        return cmds[0] if cmds else None
    
    def _evaluate_select_number_options(self, max_val: int) -> int:
        """Evaluate SELECT_NUMBER options and return the best choice.
        
        Uses different evaluation strategies based on select_number_mode:
        - 'heuristic': Fast game state heuristics
        - 'evaluator': HeuristicEvaluator scoring
        - 'mcts': Monte Carlo Tree Search (future)
        """
        if not self.gs:
            return max_val
        
        if self.select_number_mode == 'evaluator' and self.evaluator:
            return self._evaluate_with_evaluator(max_val)
        elif self.select_number_mode == 'mcts' and self.mcts:
            return self._evaluate_with_mcts(max_val)
        else:
            return self._evaluate_with_heuristic(max_val)
    
    def _evaluate_with_heuristic(self, max_val: int) -> int:
        """Fast heuristic evaluation based on game state."""
        try:
            active_pid = EngineCompat.get_active_player_id(self.gs)
            player = self.gs.players[active_pid]
            opponent = self.gs.players[1 - active_pid]
            
            # Get game state metrics
            hand_size = len(player.hand)
            deck_size = len(player.deck)
            mana_size = len(player.mana_zone)
            battle_size = len(player.battle_zone)
            opp_battle_size = len(opponent.battle_zone)
            
            # Deck-out safety: leave at least 5 cards
            safe_draw = min(max_val, max(0, deck_size - 5))
            
            # Critical situations
            if deck_size <= 5:
                return 0  # Avoid deck-out
            
            # Hand management
            if hand_size >= 7:
                return min(1, safe_draw)  # Avoid hand flood
            
            # Game phase detection
            if mana_size <= 3:
                # Early game: Prioritize resources
                return safe_draw
            elif battle_size == 0 and hand_size <= 3:
                # Need board presence
                return safe_draw
            elif opp_battle_size >= 3 and battle_size <= 1:
                # Behind on board: need answers
                return min(safe_draw, max(2, int(max_val * 0.7)))
            else:
                # Mid-game balance: ~60% of max
                return min(max(1, int(max_val * 0.6)), safe_draw)
                
        except Exception:
            return max(1, max_val // 2)
    
    def _evaluate_with_evaluator(self, max_val: int) -> int:
        """Evaluate options using HeuristicEvaluator."""
        try:
            active_pid = EngineCompat.get_active_player_id(self.gs)
            
            # Evaluate current state
            states = [self.gs]
            _, values = self.evaluator.evaluate(states)
            current_value = values[0] if values else 0.0
            
            # Use evaluator's assessment to inform decision
            # Higher value = stronger position = can afford conservative play
            # Lower value = weaker position = need aggressive draws
            
            player = self.gs.players[active_pid]
            deck_size = len(player.deck)
            hand_size = len(player.hand)
            
            safe_draw = min(max_val, max(0, deck_size - 5))
            
            if current_value > 0.3:
                # Strong position: conservative
                if hand_size >= 5:
                    return min(1, safe_draw)
                else:
                    return min(max(1, int(max_val * 0.5)), safe_draw)
            elif current_value < -0.3:
                # Weak position: aggressive
                return safe_draw
            else:
                # Balanced: moderate
                return min(max(1, int(max_val * 0.6)), safe_draw)
                
        except Exception as e:
            self.callback_log(f"Evaluator error: {e}, fallback to heuristic")
            return self._evaluate_with_heuristic(max_val)
    
    def _evaluate_with_mcts(self, max_val: int) -> int:
        """Evaluate options using lightweight simulation.
        
        Since full MCTS requires game state cloning, we use a hybrid approach:
        1. Heuristic Agent to predict likely actions after each choice
        2. Evaluator to score the resulting positions
        3. Choose the option with best expected value
        """
        try:
            if not self.evaluator:
                self.evaluator = dm_ai_module.HeuristicEvaluator(self.card_db)
            
            active_pid = EngineCompat.get_active_player_id(self.gs)
            player = self.gs.players[active_pid]
            deck_size = len(player.deck)
            
            # Safety bounds
            safe_draw = min(max_val, max(0, deck_size - 5))
            
            # Simulate depth-1 evaluation for each choice
            best_choice = 0
            best_score = float('-inf')
            
            # Try a few key options instead of all (for performance)
            options_to_try = [0, max(1, max_val // 4), max(1, max_val // 2), 
                             max(1, int(max_val * 0.75)), safe_draw]
            options_to_try = sorted(set(o for o in options_to_try if 0 <= o <= safe_draw))
            
            for choice in options_to_try:
                # Estimate value after drawing 'choice' cards
                # More cards = more hand quality but less deck
                # Score based on hand advantage vs deck risk
                
                projected_hand = len(player.hand) + choice
                projected_deck = deck_size - choice
                
                # Scoring factors
                hand_quality = min(projected_hand / 7.0, 1.0)  # Normalized to [0,1]
                deck_safety = projected_deck / max(1, deck_size)  # Remaining deck ratio
                
                # Penalty for over-drawing
                overdraw_penalty = max(0, projected_hand - 7) * 0.2
                
                # Penalty for deck danger
                if projected_deck < 5:
                    deck_penalty = (5 - projected_deck) * 0.5
                else:
                    deck_penalty = 0
                
                # Combined score
                score = hand_quality + deck_safety - overdraw_penalty - deck_penalty
                
                # Bonus for moderate draws (avoid extremes)
                if 0.3 <= choice / max(1, max_val) <= 0.7:
                    score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_choice = choice
            
            self.callback_log(f"Simulation selected {best_choice}/{safe_draw} (score={best_score:.2f})")
            return best_choice
            
        except Exception as e:
            self.callback_log(f"MCTS simulation error: {e}, fallback to heuristic")
            return self._evaluate_with_heuristic(max_val)

    def _build_default_deck(self) -> List[int]:
        """Build a default deck from card_db."""
        if not self.card_db:
            return []
        
        try:
            # Use first 40 cards from card_db
            all_ids = list(self.card_db.keys())
            return (all_ids * 40)[:40]
        except Exception:
            return []

    def generate_legal_commands(self) -> List[Any]:
        """Get legal commands from C++ engine."""
        if not self.gs:
            return []
        from dm_toolkit import commands
        return commands.generate_legal_commands(self.gs, self.card_db)

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.gs.game_over if self.gs else False

    def set_player_mode(self, player_id: int, mode: str):
        """Set player mode: 'Human' or 'AI'."""
        self.player_modes[player_id] = mode
        self.callback_log(f"P{player_id} mode set to: {mode}")

    def toggle_auto_step(self):
        """Toggle automatic game progression."""
        self.is_running = not self.is_running
        if self.is_running:
            self.callback_log("Auto-step enabled")
            self._auto_step_loop()
        else:
            self.callback_log("Auto-step disabled")

    def _auto_step_loop(self):
        """Continuously step game until disabled or game over."""
        if not self.is_running or self.is_game_over():
            self.is_running = False
            return
        
        self.step_game()
        
        if self.is_running:
            # Schedule next step
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self._auto_step_loop)
            except Exception:
                self.is_running = False

    def _check_and_handle_input_wait(self) -> bool:
        """Check if waiting for user input and handle input request."""
        if not self.gs:
            return False
        
        try:
            waiting = getattr(self.gs, 'waiting_for_user_input', False)
        except Exception:
            waiting = False
        
        if not waiting:
            return False

        # Stop auto-stepping if running
        if self.is_running:
            self.is_running = False

        # Trigger input request callback
        if self.callback_input_request:
            try:
                self.callback_input_request()
            except Exception:
                pass

        self.callback_update_ui()
        return True

    def resume_from_input(self, result: Any):
        """Resume game after user input (for effect resolution)."""
        if not self.gs or not dm_ai_module:
            return
        
        try:
            if hasattr(dm_ai_module, 'EffectResolver') and hasattr(dm_ai_module.EffectResolver, 'resume'):
                dm_ai_module.EffectResolver.resume(self.gs, self.card_db, result)
                self._fast_forward()
        except Exception as e:
            self.callback_log(f"Resume error: {e}")
        
        self.callback_update_ui()
