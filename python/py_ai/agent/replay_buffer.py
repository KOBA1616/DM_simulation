import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000, golden_ratio=0.3):
        self.capacity = capacity
        self.golden_ratio = golden_ratio
        self.buffer = deque(maxlen=int(capacity * (1 - golden_ratio)))
        self.golden_buffer = deque(maxlen=int(capacity * golden_ratio))
        
    def push(self, game_data, is_golden=False):
        """
        game_data: list of (state_tensor, policy_target, value_target)
        """
        if is_golden:
            self.golden_buffer.extend(game_data)
        else:
            self.buffer.extend(game_data)
            
    def sample(self, batch_size):
        total_len = len(self.buffer) + len(self.golden_buffer)
        if total_len < batch_size:
            return None
            
        # Mix samples
        # Simple random sampling from both
        # Or strictly follow ratio?
        # Let's just sample from combined list for simplicity now, 
        # or sample proportionally if we want to enforce ratio in batch.
        
        # Efficient sampling:
        # We can't easily sample from deque without converting to list.
        # But converting large deque to list is slow.
        # Usually we use a fixed size list with pointer.
        # For this prototype, let's just use random.sample on list(deque) which is slow but works.
        # Optimization: Use list and replace random indices.
        
        combined = list(self.buffer) + list(self.golden_buffer)
        batch = random.sample(combined, batch_size)
        
        state_batch = []
        policy_batch = []
        value_batch = []
        
        for s, p, v in batch:
            state_batch.append(s)
            policy_batch.append(p)
            value_batch.append(v)
            
        return (
            np.array(state_batch),
            np.array(policy_batch),
            np.array(value_batch)
        )
        
    def __len__(self):
        return len(self.buffer) + len(self.golden_buffer)
