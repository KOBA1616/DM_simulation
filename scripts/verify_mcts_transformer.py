
import sys
import os
import time
from unittest.mock import MagicMock, Mock

# Add root to path
sys.path.insert(0, os.getcwd())

# Mock numpy
try:
    import numpy as np
except ImportError:
    print("Numpy not found. Mocking numpy.")
    np = MagicMock()
    np.random.dirichlet = lambda alpha: [1.0/len(alpha)] * len(alpha)
    np.array = lambda x, dtype=None: x
    np.issubdtype = lambda x, y: True # Assume int check passes
    np.integer = int
    sys.modules['numpy'] = np

# Mock torch
try:
    import torch
    print("Torch found.")
except ImportError:
    print("Torch not found. Mocking torch.")
    torch = MagicMock()
    torch.Tensor = MagicMock
    torch.float32 = 'float32'
    torch.long = 'long'
    def mock_tensor(data, dtype=None):
        m = MagicMock()
        if isinstance(data, list):
            m.shape = (1, len(data))
            m.dim.return_value = 1
            m.numpy.return_value = data
        else:
            m.shape = (1, 1)
            m.dim.return_value = 0
        m.unsqueeze.return_value = m
        m.dtype = dtype
        return m
    torch.tensor = mock_tensor

    # Create simple classes for NN modules to avoid MagicMock spec issues
    class MockModule:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs):
            # Forward to self.forward to simulate nn.Module behavior
            return self.forward(*args, **kwargs)
        def forward(self, *args, **kwargs):
            return MagicMock()
        def parameters(self): return []

    torch.nn = MagicMock()
    torch.nn.Module = MockModule
    torch.nn.Embedding = MockModule
    torch.nn.TransformerEncoderLayer = MockModule
    torch.nn.TransformerEncoder = MockModule
    torch.nn.LayerNorm = MockModule
    torch.nn.Linear = MockModule
    torch.nn.GELU = MockModule
    torch.nn.Sequential = MockModule

    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = MagicMock()

# Import modules
try:
    import dm_ai_module
    from dm_toolkit.ai.agent.mcts import MCTS
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
except ImportError as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

def verify_integration():
    print("--- Verifying MCTS + Transformer + TokenConverter Integration ---")

    # 1. Setup Game
    print("Initializing GameInstance...")
    game = dm_ai_module.GameInstance()
    game.initialize_card_stats(40)
    game.start_game()
    state = game.state

    # Verify Phase Manager basic setup
    print(f"Initial Phase: {state.current_phase} (Expected 2=MANA)")

    # 2. Setup Model (Mocked or Real)
    print("Initializing DuelTransformer...")
    model = DuelTransformer(vocab_size=1000, action_dim=100)

    # Mock the forward pass to return policy/value
    model.forward = MagicMock(return_value=(torch.tensor([0.1]*100), torch.tensor([0.5])))
    # Note: model.__call__ is now handled by MockModule base class calling model.forward

    # 3. Setup Converter
    print("Setting up TokenConverter...")
    if hasattr(dm_ai_module, 'TokenConverter'):
        def converter_wrapper(s, p, db):
            return dm_ai_module.TokenConverter.encode_state(s, p)
    else:
        print("ERROR: TokenConverter still not found in dm_ai_module!")
        return

    # 4. Setup MCTS
    print("Initializing MCTS...")
    mcts = MCTS(
        network=model,
        card_db={},
        simulations=5,
        state_converter=converter_wrapper
    )

    # 5. Run Search (Verification of Type Safety)
    print("Running MCTS Search...")
    try:
        # Mock legal actions to avoid engine calls if pure fallback is incomplete
        # dm_ai_module.ActionGenerator is available though.
        root_node = mcts.search(state)
        print("MCTS Search completed successfully.")
    except Exception as e:
        print(f"MCTS Search FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Verify Input Tensor Type
    if model.forward.called:
        args = model.forward.call_args[0]
        input_tensor = args[0]
        print(f"Model received input tensor: {input_tensor}")
        print(f"Input tensor dtype: {getattr(input_tensor, 'dtype', 'Unknown')}")

        # Verify it used Long (token) type
        if getattr(input_tensor, 'dtype', None) == torch.long:
            print("SUCCESS: Input tensor is Long (Int) type.")
        else:
            print("WARNING: Input tensor dtype mismatch (Expected Long).")
            # In mock, we set 'long' string.
            if getattr(input_tensor, 'dtype', None) == 'long':
                 print("SUCCESS: Input tensor is Long (Mocked).")
    else:
        print("Model was NOT called!")

    # 7. Verify Phase Transitions
    print("\n--- Verifying Phase Transitions ---")
    pm = dm_ai_module.PhaseManager

    print(f"Start: Phase={state.current_phase}, P{state.active_player_id}, Turn={state.turn_number}")

    # MANA -> MAIN
    pm.next_phase(state, {})
    print(f"After next_phase: Phase={state.current_phase} (Expected 3=MAIN)")
    if state.current_phase != 3: print("FAIL: Did not go to MAIN")

    # MAIN -> ATTACK
    pm.next_phase(state, {})
    print(f"After next_phase: Phase={state.current_phase} (Expected 4=ATTACK)")
    if state.current_phase != 4: print("FAIL: Did not go to ATTACK")

    # ATTACK -> END
    pm.next_phase(state, {})
    print(f"After next_phase: Phase={state.current_phase} (Expected 5=END)")
    if state.current_phase != 5: print("FAIL: Did not go to END")

    # END -> MANA (Next Turn)
    old_player = state.active_player_id
    pm.next_phase(state, {})
    print(f"After next_phase: Phase={state.current_phase} (Expected 2=MANA)")
    print(f"Player: {state.active_player_id} (Expected {1-old_player})")

    if state.active_player_id == old_player: print("FAIL: Player did not switch")

    print("Phase Logic Verification Complete.")

if __name__ == "__main__":
    verify_integration()
