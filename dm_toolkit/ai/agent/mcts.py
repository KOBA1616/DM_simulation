"""
Python MCTS implementation.
Restored to support Token-based Transformer models which are incompatible with the current C++ engine's Float-based TensorConverter.
"""
import math
try:
    import torch
except ImportError:
    torch = None
import numpy as np
import dm_ai_module
from dm_toolkit import commands
from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.engine.compat import EngineCompat
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
        if torch is None:
            raise RuntimeError("MCTS requires 'torch' to be installed.")

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

    def search(self, root_state: Any, add_noise: bool = False) -> Any:
        # Returns the best action (Object)
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

        # Select best action from root
        best_child = None
        best_count = -1
        for child in root.children:
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_child = child

        return best_child.action if best_child else None

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
            
            # AlphaZero logic: Q is usually Mean Value.
            # If child state is opponent's turn, network returned "Opponent Win Prob".
            # So for us (Parent), value is -Value.
            # But _backpropagate handles this via perspective comparison.
            # node.value() (which is child.value_sum / visits) stores utility for player at child.state?
            # NO.
            # My fixed _backpropagate accumulates `value` if `node.state.active_player_id == leaf_player`.
            # Wait. `node.value_sum` accumulates utility relative to WHOM?
            #
            # In `_backpropagate`:
            # leaf_player = node (the leaf).
            # We add +value to nodes where active_player == leaf_player.
            # We add -value to nodes where active_player != leaf_player.
            # So `value_sum` at `node` represents "Utility for the player who is active at `node`".
            #
            # Example:
            # Root (P0). Child (P0) [Same Turn].
            # Leaf (P0). Value = +1 (Win for P0).
            # Backprop:
            # Leaf (P0): Match. += 1.
            # Child (P0): Match. += 1.
            # Root (P0): Match. += 1.
            #
            # Example:
            # Root (P0). Child (P1) [Turn Change].
            # Leaf (P1). Value = +1 (Win for P1).
            # Backprop:
            # Leaf (P1): Match. += 1.
            # Child (P1): Match. += 1.
            # Root (P0): Mismatch (P0 != P1). -= 1. (Bad for P0).
            #
            # So `value_sum` correctly stores utility for `node.state.active_player_id`.
            #
            # Now `_select_child`:
            # Parent is P0. Child is P0.
            # Parent wants to MAXIMIZE utility for P0.
            # Child.value() is utility for P0.
            # Score = Q + U. We MAXIMIZE. Correct.
            #
            # Parent is P0. Child is P1.
            # Parent wants to MAXIMIZE utility for P0.
            # Child.value() is utility for P1.
            # Utility for P0 = -Child.value().
            # So Score = (-Q) + U.
            #
            # So we DO need a flip in `_select_child` if player changes!

            if node.state.active_player_id != child.state.active_player_id:
                q_score = -q_score

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
            current_player = node.state.active_player_id
            # Mapping result (int or enum)
            res_val = int(result) if result is not None else 0
            if res_val == 3: # DRAW
                return 0.0
            if res_val == 1: # P1 WIN (ID 0)
                return 1.0 if current_player == 0 else -1.0
            if res_val == 2: # P2 WIN (ID 1)
                return 1.0 if current_player == 1 else -1.0
            return 0.0

        # Generate legal moves
        try:
            try:
                cmd_list = commands.generate_legal_commands(node.state, self.card_db) or []
            except Exception:
                cmd_list = []
            actions = []
            if not cmd_list:
                try:
                    # Fallback to legacy ActionGenerator
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
        iterable = actions if actions else cmd_list
        for item in iterable:
            # Map action to index when possible
            action_idx = -1

            if self.action_encoder:
                try:
                    action_idx = self.action_encoder(item, node.state, node.state.active_player_id)
                except Exception:
                    action_idx = -1

            # Fallback encoders...
            if action_idx == -1:
                 # Minimal fallback: assume uniform prior if unknown
                 pass

            # Clone state
            next_state = node.state.clone()

            # Apply action/command
            try:
                cmd = ensure_executable_command(item)
                EngineCompat.ExecuteCommand(next_state, cmd, self.card_db)
            except Exception:
                pass

            # Check pass/phase-change is handled by Engine/PhaseManager usually.
            # But we might need to nudge it if Engine doesn't auto-phase.
            # MCTS uses _fast_forward to skip empty phases.

            # Fast forward through auto-phases
            self._fast_forward(next_state)

            child = MCTSNode(next_state, parent=node, action=item)

            # Prior probability from policy (if we have an index)
            if 0 <= action_idx < len(policy):
                child.prior = float(policy[action_idx])
            else:
                child.prior = 1e-6

            node.children.append(child)

        return value

    def _backpropagate(self, node: Optional[MCTSNode], value: float) -> None:
        # Value is initially from perspective of the player at the LEAF node.
        # We assume the leaf node has just been expanded and its state evaluated.
        # So 'value' is Utility(Leaf.ActivePlayer).
        if node is None:
            return

        leaf_player = node.state.active_player_id

        while node is not None:
            node.visit_count += 1
            if node.state.active_player_id == leaf_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            node = node.parent
