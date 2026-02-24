# -*- coding: utf-8 -*-
import math
import random
import dm_ai_module
from dm_toolkit import commands as commands
from PyQt6.QtWidgets import QApplication

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this state
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = []
        self.player_id = state.active_player_id

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.value_sum / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

from typing import Callable, Optional

class PythonMCTS:
    def __init__(self, card_db, simulations=100):
        self.card_db = card_db
        self.simulations = simulations
        self.root = None
        self.should_stop: Optional[Callable[[], bool]] = None # Callback function

    def search(self, root_state):
        self.root = Node(root_state.clone())
        # Compatibility: obtain both legacy Action objects (when available)
        # and ICommand wrappers (command-first path) for execution.
        # Prefer command-first generator, fallback to legacy ActionGenerator
        try:
            legal_commands = []
            try:
                legal_commands = commands.generate_legal_commands(self.root.state, self.card_db, strict=False, skip_wrapper=True) or []
            except TypeError:
                legal_commands = commands.generate_legal_commands(self.root.state, self.card_db) or []
            except Exception:
                legal_commands = []
            legal_actions = []
            if not legal_commands:
                try:
                    legal_actions = commands.generate_legal_commands(self.root.state, self.card_db, strict=False, skip_wrapper=True) or []
                except Exception:
                    try:
                        legal_actions = commands.generate_legal_commands(self.root.state, self.card_db) or []
                    except Exception:
                        legal_actions = []
        except Exception:
            legal_actions = []
            legal_commands = []

        def _is_pass(obj):
            # Detect PASS for either Action or Command
            try:
                if hasattr(obj, 'type'):
                    return getattr(obj, 'type') == dm_ai_module.CommandType.PASS
            except Exception:
                pass
            try:
                # command_new returns objects with to_dict or payload
                if hasattr(obj, 'to_dict'):
                    d = obj.to_dict() or {}
                    if d.get('kind') == 'FlowCommand' and d.get('payload') is None:
                        pass
                # try payload attribute
                payload = getattr(obj, 'payload', None)
                if isinstance(payload, dict) and payload.get('pass'):
                    return True
            except Exception:
                pass
            return False

        # Rule: Always charge mana until turn 5
        if self.root.state.current_phase == dm_ai_module.Phase.MANA and self.root.state.turn_number <= 5:
            has_charge = any((getattr(a, 'type', None) == dm_ai_module.CommandType.MANA_CHARGE) for a in legal_actions) or any((getattr(c, 'payload', {}).get('add_mana') is not None) for c in legal_commands)
            if has_charge:
                # Remove PASS action to force charge
                legal_actions = [a for a in legal_actions if not (getattr(a, 'type', None) == dm_ai_module.CommandType.PASS)]
                # also filter commands similarly by payload
                legal_commands = [c for c in legal_commands if not (getattr(c, 'payload', {}).get('pass'))]

        # Rule: Prioritize playing cards in Main Phase (80% chance to force play if possible)
        elif self.root.state.current_phase == dm_ai_module.Phase.MAIN:
            has_play = any((getattr(a, 'type', None) == dm_ai_module.CommandType.PLAY_CARD) for a in legal_actions) or any((getattr(c, 'payload', {}).get('play')) for c in legal_commands)
            if has_play and random.random() < 0.8:
                 legal_actions = [a for a in legal_actions if not (getattr(a, 'type', None) == dm_ai_module.CommandType.PASS)]
                 legal_commands = [c for c in legal_commands if not (getattr(c, 'payload', {}).get('pass'))]

        # Rule: Prioritize attacking in Attack Phase (80% chance to force attack if possible)
        elif self.root.state.current_phase == dm_ai_module.Phase.ATTACK:
            has_attack = any((getattr(a, 'type', None) in (dm_ai_module.CommandType.ATTACK_PLAYER, dm_ai_module.CommandType.ATTACK_CREATURE)) for a in legal_actions) or any((getattr(c, 'payload', {}).get('attack')) for c in legal_commands)
            if has_attack and random.random() < 0.8:
                 legal_actions = [a for a in legal_actions if not (getattr(a, 'type', None) == dm_ai_module.CommandType.PASS)]
                 legal_commands = [c for c in legal_commands if not (getattr(c, 'payload', {}).get('pass'))]

        # Prefer to store Action objects (for EffectResolver compatibility); if absent, store command objects
        if legal_actions:
            self.root.untried_actions = legal_actions
        else:
            self.root.untried_actions = legal_commands

        # Check stop condition before starting
        if self.should_stop and self.should_stop():
            return None

        for _ in range(self.simulations):
            QApplication.processEvents() # Keep GUI responsive
            if self.should_stop and self.should_stop():
                break
            
            node = self._select(self.root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        if self.should_stop and self.should_stop():
            return None

        if not self.root.children:
            if self.root.untried_actions:
                return random.choice(self.root.untried_actions)
            return None # Should be pass

        return self.root.best_child(c_param=0.0).action

    def _select(self, node):
        while not self._is_terminal(node):
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                if not node.children:
                    # Node is fully expanded (no untried actions) but has no children.
                    # This means there were no legal actions at all in this state.
                    return node
                node = node.best_child()
        return node

    def _expand(self, node):
        action = node.untried_actions.pop()
        next_state = node.state.clone()
        # Support both Action and ICommand objects
        try:
            from dm_toolkit.unified_execution import ensure_executable_command
            from dm_toolkit.engine.compat import EngineCompat
            if hasattr(action, 'type'):
                cmd = ensure_executable_command(action)
                EngineCompat.ExecuteCommand(next_state, cmd, self.card_db)
            else:
                # ICommand-like
                try:
                    if hasattr(next_state, 'execute_command'):
                        next_state.execute_command(action)
                    elif hasattr(action, 'execute'):
                        action.execute(next_state)
                    else:
                        EngineCompat.ExecuteCommand(next_state, action, self.card_db)
                except Exception:
                    EngineCompat.ExecuteCommand(next_state, action, self.card_db)
        except Exception:
            pass

        # Check if phase changed; detect PASS/MANA_CHARGE both for Action and Command
        is_pass_action = False
        try:
            if hasattr(action, 'type'):
                is_pass_action = (getattr(action, 'type') == dm_ai_module.CommandType.PASS or getattr(action, 'type') == dm_ai_module.CommandType.MANA_CHARGE)
            else:
                payload = getattr(action, 'payload', {}) or {}
                is_pass_action = bool(payload.get('pass') or payload.get('add_mana'))
        except Exception:
            is_pass_action = False

        if is_pass_action:
             dm_ai_module.PhaseManager.next_phase(next_state, self.card_db)

        child_node = Node(next_state, parent=node, action=action)
        # Populate untried actions for child (prefer Action list)
            try:
                # Prefer command-first generator, fallback to legacy ActionGenerator only when needed
                child_cmds = []
                try:
                    child_cmds = commands.generate_legal_commands(next_state, self.card_db, strict=False, skip_wrapper=True) or []
                except TypeError:
                    child_cmds = commands.generate_legal_commands(next_state, self.card_db) or []
                except Exception:
                    child_cmds = []
                child_actions = []
                    if not child_cmds:
                    try:
                        child_actions = commands.generate_legal_commands(next_state, self.card_db, strict=False, skip_wrapper=True) or []
                    except Exception:
                        try:
                            child_actions = commands.generate_legal_commands(next_state, self.card_db) or []
                        except Exception:
                            child_actions = []
                child_node.untried_actions = child_cmds if child_cmds else child_actions
            except Exception:
                child_node.untried_actions = []
        node.children.append(child_node)
        return child_node

    def _simulate(self, state):
        current_state = state.clone()
        depth = 0
        while depth < 20: # Max depth to prevent infinite loops
            is_over, result = dm_ai_module.PhaseManager.check_game_over(current_state)
            if is_over:
                # Result: 0=P0 Win, 1=P1 Win, 2=Draw
                # We want reward for the root player.
                # Assuming root player is active_player_id of root state.
                if self.root is None:
                    return 0.0
                root_player = self.root.player_id
                if result == root_player:
                    return 1.0
                elif result == 2: # Draw
                    return 0.5
                else:
                    return 0.0
            
                try:
                    # Prefer command-first during simulation rollouts
                    actions = []
                    try:
                        actions = commands.generate_legal_commands(current_state, self.card_db, strict=False, skip_wrapper=True) or []
                    except TypeError:
                        actions = commands.generate_legal_commands(current_state, self.card_db) or []
                    except Exception:
                        actions = []
                    if not actions:
                        try:
                            actions = commands.generate_legal_commands(current_state, self.card_db, strict=False, skip_wrapper=True) or []
                        except Exception:
                            try:
                                actions = commands.generate_legal_commands(current_state, self.card_db) or []
                            except Exception:
                                actions = []
                except Exception:
                    actions = []
            if not actions:
                dm_ai_module.PhaseManager.next_phase(current_state, self.card_db)
            else:
                # Heuristics for Random Rollout
                
                # 1. Mana Charge (Turn <= 3)
                mana_charges = [a for a in actions if a.type == dm_ai_module.CommandType.MANA_CHARGE]
                should_charge = False
                if mana_charges:
                    if current_state.turn_number <= 3:
                        should_charge = True
                    elif random.random() < 0.9: 
                        should_charge = True
                
                if should_charge:
                    action = random.choice(mana_charges)
                else:
                    # 2. Play Card (Main Phase)
                    play_cards = [a for a in actions if a.type == dm_ai_module.CommandType.PLAY_CARD]
                    if play_cards and random.random() < 0.8: # 80% chance to play card
                        action = random.choice(play_cards)
                    else:
                        # 3. Attack (Attack Phase)
                        attacks = [a for a in actions if a.type in (dm_ai_module.CommandType.ATTACK_PLAYER, dm_ai_module.CommandType.ATTACK_CREATURE)]
                        if attacks and random.random() < 0.8: # 80% chance to attack
                            action = random.choice(attacks)
                        else:
                            # Fallback to random (includes PASS)
                            action = random.choice(actions)
                
                try:
                    from dm_toolkit.unified_execution import ensure_executable_command
                    from dm_toolkit.engine.compat import EngineCompat
                    cmd = ensure_executable_command(action)
                    EngineCompat.ExecuteCommand(current_state, cmd, self.card_db)
                except Exception:
                    try:
                        from dm_toolkit.compat_wrappers import execute_action_compat
                        execute_action_compat(current_state, action, self.card_db)
                    except Exception:
                        try:
                            # Last resort: call native EffectResolver
                            dm_ai_module.EffectResolver.resolve_action(current_state, action, self.card_db)
                        except Exception:
                            pass
                # Phase advancement: detect both Action and ICommand representations
                try:
                    if hasattr(action, 'type') and getattr(action, 'type', None) in (dm_ai_module.CommandType.PASS, dm_ai_module.CommandType.MANA_CHARGE):
                        dm_ai_module.PhaseManager.next_phase(current_state, self.card_db)
                    else:
                        payload = getattr(action, 'payload', {}) or {}
                        if payload.get('pass') or payload.get('add_mana'):
                            dm_ai_module.PhaseManager.next_phase(current_state, self.card_db)
                except Exception:
                    pass
            depth += 1
        
        return 0.5 # Draw if depth limit reached

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            # If node's player is the one who won, add reward.
            # But MCTS usually maximizes for the player at that node.
            # Standard MCTS:
            # If root is P0. Child is P1's move.
            # If P0 wins, reward is 1.
            # Node P1 should see this as bad?
            # Actually, usually we alternate perspectives.
            # If result is win for root_player (P0).
            # Node (P0's turn): Value += 1
            # Node (P1's turn): Value += 0 (or -1)
            
            # Simplified: Always view from root player's perspective?
            # No, Minimax style.
            # But here we just accumulate reward for the player who made the move?
            # Let's stick to: reward is 1.0 if root_player wins.
            # If node.player_id == root_player, add reward.
            # If node.player_id != root_player, add 1-reward?
            
            # Let's use simple "Win for Root Player" metric.
            # If the result was a win for root player, we add 1 to nodes where it was root player's turn?
            # Wait, standard UCT assumes alternating max if we flip rewards, or max/min.
            # Let's assume 2-player zero-sum.
            
            if self.root is None:
                return

            root_player = self.root.player_id
            # Reward is 1.0 if root_player won.
            
            # If it was P0's turn (node.player_id == P0), and P0 won, that's good.
            # If it was P1's turn, and P0 won, that's bad for P1.
            
            if node.player_id == root_player:
                node.value_sum += reward
            else:
                node.value_sum += (1.0 - reward)
                
            node = node.parent

    def _is_terminal(self, node):
        is_over, _ = dm_ai_module.PhaseManager.check_game_over(node.state)
        return is_over

    def get_tree_data(self):
        # Convert tree to dictionary format for visualization
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node):
        if not node:
            return {}
        
        data = {
            "name": node.action.to_string() if node.action else "Root",
            "visits": node.visits,
            "value": node.value_sum,
            "children": []
        }
        
        # Sort children by visits
        sorted_children = sorted(node.children, key=lambda c: c.visits, reverse=True)
        
        for child in sorted_children:
            data["children"].append(self._node_to_dict(child))
            
        return data
