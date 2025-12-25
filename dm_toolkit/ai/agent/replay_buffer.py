import random
import numpy as np
from typing import Any, List, Tuple, Optional


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, golden_ratio: float = 0.2) -> None:
        self.capacity: int = capacity
        self.golden_capacity: int = int(capacity * golden_ratio)
        self.regular_capacity: int = capacity - self.golden_capacity

        self.regular_buffer: List[Tuple[Any, Any, Any]] = []
        self.golden_buffer: List[Tuple[Any, Any, Any]] = []
        self.regular_idx: int = 0
        self.golden_idx: int = 0
        
    def push(self, game_data: List[Tuple[Any, Any, Any]], is_golden: bool = False) -> None:
        """
        game_data: list of (state_tensor, policy_target, value_target)
        """
        target_buffer = self.golden_buffer if is_golden else self.regular_buffer
        target_cap = self.golden_capacity if is_golden else self.regular_capacity
        
        # If we receive a batch of data, we can extend.
        # But to maintain a circular buffer efficiently with lists, 
        # we usually just append and then slice, or overwrite.
        # For simplicity and performance in Python:
        # Just append. If len > cap, remove from beginning.
        # This is O(k) where k is amount added.
        
        target_buffer.extend(game_data)
        
        excess = len(target_buffer) - target_cap
        if excess > 0:
            # Slicing is O(N) copy, but N is capacity. 
            # Doing this every step is okay if batch is large enough.
            # Better: del target_buffer[:excess]
            del target_buffer[:excess]
            
    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        total_len = len(self.regular_buffer) + len(self.golden_buffer)
        if total_len < batch_size:
            return None
            
        # Determine how many samples from golden buffer
        # Try to respect the ratio in the batch, or just sample uniformly from available data?
        # AlphaZero usually samples uniformly from the window.
        # "Golden Games" implies we want to keep them longer, not necessarily sample them more frequently per se,
        # but if they stay longer, they get sampled more over time.
        
        # Let's sample uniformly from the union.
        # To avoid concatenating lists (O(N)), we pick indices.
        
        n_regular = len(self.regular_buffer)
        n_golden = len(self.golden_buffer)
        
        indices = np.random.randint(0, n_regular + n_golden, size=batch_size)
        
        state_batch = []
        policy_batch = []
        value_batch = []
        
        for idx in indices:
            if idx < n_regular:
                s, p, v = self.regular_buffer[idx]
            else:
                s, p, v = self.golden_buffer[idx - n_regular]
                
            state_batch.append(s)
            policy_batch.append(p)
            value_batch.append(v)
            
        return (
            np.array(state_batch),
            np.array(policy_batch),
            np.array(value_batch)
        )
        
    def __len__(self) -> int:
        return len(self.regular_buffer) + len(self.golden_buffer)
