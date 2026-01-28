import math
import torch
import numpy as np
import dm_ai_module
from dm_toolkit import commands
from typing import Any, Optional, List, Dict, Tuple, Callable


class MCTSNode:
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Any = None) -> None:
        self.state: Any = state
        self.parent: Optional['MCTSNode'] = parent
        self.action: Any = action  # Action that led to this state
        self.children: List['MCTSNode'] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network: Any, card_db: Dict[str, Any], simulations: int = 100, c_puct: float = 1.0, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25, state_converter: Optional[Callable[[Any, int, Dict], Any]] = None, action_encoder: Optional[Callable[[Any, Any, int], int]] = None) -> None:
        self.network: Any = network
        self.card_db: Dict[str, Any] = card_db
        self.simulations: int = simulations
        self.c_puct: float = c_puct
        self.dirichlet_alpha: float = dirichlet_alpha
        self.dirichlet_epsilon: float = dirichlet_epsilon
        self.state_converter = state_converter
        self.action_encoder = action_encoder

    def _fast_forward(self, state: Any) -> None:
        dm_ai_module.PhaseManager.fast_forward(state, self.card_db)

    def search(self, root_state: Any, add_noise: bool = False) -> MCTSNode:
        root_state_clone = root_state.clone()
        self._fast_forward(root_state_clone)
        root = MCTSNode(root_state_clone)

        # Expand root
        self._expand(root)
        
        if add_noise:
            self._add_exploration_noise(root)

        for _ in range(self.simulations):
            node = root

            # Selection
            while node.is_expanded():
                next_node = self._select_child(node)
                if next_node is None:
                    break
                node = next_node

            # Expansion & Evaluation
            if not node.is_expanded():
                value = self._expand(node)
            else:
                value = node.value()

            # Backpropagation
            self._backpropagate(node, value)

        return root

    def _add_exploration_noise(self, node: MCTSNode) -> None:
        if not node.children:
            return
            
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        
        for i, child in enumerate(node.children):
            child.prior = child.prior * (1 - self.dirichlet_epsilon) + noise[i] * self.dirichlet_epsilon

    def _select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            u_score = self.c_puct * child.prior * math.sqrt(
                node.visit_count
            ) / (1 + child.visit_count)
            q_score = child.value()
            # If active player is different, we might need to flip value?
            # AlphaZero usually assumes value is for the current player.
            # If next state is opponent's turn, the value returned by network
            # for that state is for opponent.
            # So Q should be -Value(child_state).
            # But here we store value_sum.
            # Let's assume standard AlphaZero: Q is mean value.
            # If child state is opponent's turn, the value we get from network
            # is "Probability Opponent Wins".
            # So for us (Parent), it is -Value.

            score = q_score + u_score
            
            if math.isnan(score):
                score = -float('inf')

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None and node.children:
            best_child = node.children[0]

        return best_child

    def _expand(self, node: MCTSNode) -> float:
        # Check game over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(node.state)
        if is_over:
            # Result: 0=None, 1=P1_WIN, 2=P2_WIN, 3=DRAW
            # We need value for the player who just moved
            # (node.parent.state.active_player)
            # Or simply, value for the player at node.state.active_player_id?
            # Usually value is [-1, 1] from perspective of current player.
            # If P1 wins, and current is P1, value = 1.
            # If P1 wins, and current is P2, value = -1.

            current_player = node.state.active_player_id
            if result == 3:
                return 0.0
            if result == 1:
                return 1.0 if current_player == 0 else -1.0
            if result == 2:
                return 1.0 if current_player == 1 else -1.0
            return 0.0

        # Generate legal moves: prefer command-first (`ICommand` wrappers), fallback to legacy Actions
        try:
            try:
                cmd_list = commands.generate_legal_commands(node.state, self.card_db) or []
            except Exception:
                cmd_list = []
            actions = []
            if not cmd_list:
                try:
                    # Fallback to legacy ActionGenerator.generate_legal_actions
                    actions = dm_ai_module.ActionGenerator.generate_legal_commands(node.state, self.card_db) or []
                except Exception:
                    actions = []
            if not cmd_list and not actions:
                return 0.0
        except Exception:
            return 0.0

        # Evaluate with Network
        # Use Masked Tensor (mask_opponent_hand=True) during inference
        if self.state_converter:
            tensor = self.state_converter(
                node.state, node.state.active_player_id, self.card_db
            )
            # Check if tensor is already a tensor or numpy array of ints (tokens)
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 1:
                    tensor_t = tensor.unsqueeze(0)
                else:
                    tensor_t = tensor
            elif isinstance(tensor, (list, np.ndarray)) and (
                (isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], (int, np.integer))) or
                (isinstance(tensor, np.ndarray) and np.issubdtype(tensor.dtype, np.integer))
            ):
                tensor_t = torch.tensor(tensor, dtype=torch.long).unsqueeze(0)
            else:
                 # Fallback/Default float tensor
                tensor_t = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
        else:
            tensor = dm_ai_module.TensorConverter.convert_to_tensor(
                node.state, node.state.active_player_id, self.card_db, True
            )
            tensor_t = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.network(tensor_t)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
        value = float(value.item())

        # Create children
        # Iterate over Action objects if present, otherwise over ICommand objects
        iterable = actions if actions else cmd_list
        for item in iterable:
            # Identify legacy Action instances explicitly when possible
            try:
                ActionCls = getattr(dm_ai_module, 'Action', None)
                is_action = isinstance(item, ActionCls) if ActionCls is not None else False
            except Exception:
                is_action = False

            # Map action to index when possible
            action_idx = -1

            if self.action_encoder:
                try:
                    action_idx = self.action_encoder(item, node.state, node.state.active_player_id)
                except Exception:
                    action_idx = -1

            if action_idx == -1:
                # Prefer command-first encoder when possible
                try:
                    # If item is a plain dict (command-like)
                    if isinstance(item, dict):
                        action_idx = dm_ai_module.CommandEncoder.command_to_index(item)
                    else:
                        # If wrapper exposes normalized dict via `to_dict`
                        to_dict = getattr(item, 'to_dict', None)
                        if callable(to_dict):
                            try:
                                action_idx = dm_ai_module.CommandEncoder.command_to_index(item.to_dict())
                            except Exception:
                                action_idx = -1
                        else:
                            action_idx = -1
                except Exception:
                    action_idx = -1

                # Fallback to legacy ActionEncoder (supports legacy Action objects or underlying `_action`)
                if action_idx == -1:
                    try:
                        if is_action:
                            action_idx = dm_ai_module.ActionEncoder.action_to_index(item)
                        else:
                            underlying = getattr(item, '_action', None)
                            if underlying is not None:
                                action_idx = dm_ai_module.ActionEncoder.action_to_index(underlying)
                            else:
                                action_idx = -1
                    except Exception:
                        action_idx = -1

            # Clone state
            next_state = node.state.clone()

            # Apply action/command
            try:
                from dm_toolkit.unified_execution import ensure_executable_command
                from dm_toolkit.engine.compat import EngineCompat
                if is_action:
                    cmd = ensure_executable_command(item)
                    EngineCompat.ExecuteCommand(next_state, cmd, self.card_db)
                else:
                    # ICommand-like
                    if hasattr(next_state, 'execute_command'):
                        next_state.execute_command(item)
                    elif hasattr(item, 'execute'):
                        item.execute(next_state)
                    else:
                        EngineCompat.ExecuteCommand(next_state, item, self.card_db)
            except Exception:
                pass

            # Check pass/phase-change for both kinds
            try:
                if is_action and getattr(item, 'type', None) == dm_ai_module.ActionType.PASS:
                    dm_ai_module.PhaseManager.next_phase(next_state, self.card_db)
                elif not is_action:
                    payload = getattr(item, 'payload', {}) or {}
                    if payload.get('pass') or payload.get('add_mana'):
                        dm_ai_module.PhaseManager.next_phase(next_state, self.card_db)
            except Exception:
                pass

            # Fast forward through auto-phases
            self._fast_forward(next_state)

            child = MCTSNode(next_state, parent=node, action=item)

            # Prior probability from policy (if we have an index)
            if 0 <= action_idx < len(policy):
                child.prior = float(policy[action_idx])
            else:
                child.prior = 0.0

            node.children.append(child)

        return value

    def _backpropagate(self, node: Optional[MCTSNode], value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent
