import math
import numpy as np
import torch
import dm_ai_module

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action # Action that led to this state
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, network, card_db, simulations=100, c_puct=1.0):
        self.network = network
        self.card_db = card_db
        self.simulations = simulations
        self.c_puct = c_puct

    def _fast_forward(self, state):
        while True:
            is_over, _ = dm_ai_module.PhaseManager.check_game_over(state)
            if is_over:
                break
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
            if actions:
                break
            dm_ai_module.PhaseManager.next_phase(state)

    def search(self, root_state):
        root_state_clone = root_state.clone()
        self._fast_forward(root_state_clone)
        root = MCTSNode(root_state_clone)
        
        # Expand root
        self._expand(root)
        
        for _ in range(self.simulations):
            node = root
            
            # Selection
            while node.is_expanded():
                node = self._select_child(node)
                
            # Expansion & Evaluation
            value = self._expand(node)
            
            # Backpropagation
            self._backpropagate(node, value)
            
        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            u_score = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            q_score = child.value()
            # If active player is different, we might need to flip value?
            # AlphaZero usually assumes value is for the current player.
            # If next state is opponent's turn, the value returned by network for that state is for opponent.
            # So Q should be -Value(child_state).
            # But here we store value_sum.
            # Let's assume standard AlphaZero: Q is mean value.
            # If child state is opponent's turn, the value we get from network is "Probability Opponent Wins".
            # So for us (Parent), it is -Value.
            
            score = q_score + u_score
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _expand(self, node):
        # Check game over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(node.state)
        if is_over:
            # Result: 0=None, 1=P1_WIN, 2=P2_WIN, 3=DRAW
            # We need value for the player who just moved (node.parent.state.active_player)
            # Or simply, value for the player at node.state.active_player_id?
            # Usually value is [-1, 1] from perspective of current player.
            # If P1 wins, and current is P1, value = 1.
            # If P1 wins, and current is P2, value = -1.
            
            current_player = node.state.active_player_id
            if result == 3: return 0.0
            if result == 1: return 1.0 if current_player == 0 else -1.0
            if result == 2: return 1.0 if current_player == 1 else -1.0
            return 0.0

        # Generate legal actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(node.state, self.card_db)
        
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
        tensor = dm_ai_module.TensorConverter.convert_to_tensor(node.state, node.state.active_player_id)
        tensor_t = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.network(tensor_t)
            
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
        value = value.item()
        
        # Create children
        for action in actions:
            # Map action to index
            action_idx = dm_ai_module.ActionEncoder.action_to_index(action)
            
            # Clone state
            next_state = node.state.clone()
            # Apply action
            dm_ai_module.EffectResolver.resolve_action(next_state, action, self.card_db)
            # Check if phase changed?
            # If action was PASS, phase changes.
            if action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(next_state)
            
            # Fast forward through auto-phases
            self._fast_forward(next_state)
            
            child = MCTSNode(next_state, parent=node, action=action)
            
            # Prior probability from policy
            if action_idx >= 0 and action_idx < len(policy):
                child.prior = float(policy[action_idx])
            else:
                child.prior = 0.0 # Should not happen if valid
                
            node.children.append(child)
            
        return value

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value # Flip for opponent
            node = node.parent
