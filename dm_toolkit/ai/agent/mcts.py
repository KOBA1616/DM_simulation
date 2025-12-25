import math
import torch
import numpy as np
import dm_ai_module
from typing import Any, Optional, List, Dict, Tuple


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
    def __init__(self, network: Any, card_db: Dict[str, Any], simulations: int = 100, c_puct: float = 1.0, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25) -> None:
        self.network: Any = network
        self.card_db: Dict[str, Any] = card_db
        self.simulations: int = simulations
        self.c_puct: float = c_puct
        self.dirichlet_alpha: float = dirichlet_alpha
        self.dirichlet_epsilon: float = dirichlet_epsilon

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

        # Generate legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(
            node.state, self.card_db
        )

        if not actions:
            # No actions (Pass or Auto-transition?)
            # If empty, maybe we should just step phase?
            # But ActionGenerator should return PASS if allowed.
            # If truly empty, it's a bug or auto-step.
            # Let's assume we step phase and continue expansion?
            # Or treat as terminal?
            # For now, return 0.
            return 0.0

        # Evaluate with Network
        # Use Masked Tensor (mask_opponent_hand=True) during inference
        tensor = dm_ai_module.TensorConverter.convert_to_tensor(
            node.state, node.state.active_player_id, self.card_db, True
        )
        tensor_t = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.network(tensor_t)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
        value = float(value.item())

        # Create children
        for action in actions:
            # Map action to index
            action_idx = dm_ai_module.ActionEncoder.action_to_index(action)

            # Clone state
            next_state = node.state.clone()
            # Apply action
            dm_ai_module.EffectResolver.resolve_action(
                next_state, action, self.card_db
            )
            # Check if phase changed?
            # If action was PASS, phase changes.
            if action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(next_state, self.card_db)

            # Fast forward through auto-phases
            self._fast_forward(next_state)

            child = MCTSNode(next_state, parent=node, action=action)

            # Prior probability from policy
            if action_idx >= 0 and action_idx < len(policy):
                child.prior = float(policy[action_idx])
            else:
                child.prior = 0.0  # Should not happen if valid

            node.children.append(child)

        return value

    def _backpropagate(self, node: Optional[MCTSNode], value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent

