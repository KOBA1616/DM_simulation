
import sys
import os
import pytest
import time

# Add native module path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'bin', 'Release'))

try:
    import dm_ai_module as dm
except ImportError as e:
    pytest.fail(f"Could not import dm_ai_module: {e}")

def test_ai_selection():
    # Load cards
    dm.CardRegistry.load_from_json(os.path.join(PROJECT_ROOT, "data/cards.json"))
    
    # Setup Game
    state = dm.GameState()
    state.setup_test_duel()
    
    # Register test cards
    # ID 1000: High Value Card (Power 10000)
    # ID 1001: Low Value Card (Power 1000)
    
    # Define Effect for dummy cards (none)
    effects = []
    
    # High Value
    c1 = dm.CardData(1000, "HighValue", 5, dm.Civilization.FIRE, 10000, dm.CardType.CREATURE, [], effects)
    dm.CardRegistry.register_card_data(c1)
    
    # Low Value
    c2 = dm.CardData(1001, "LowValue", 1, dm.Civilization.FIRE, 1000, dm.CardType.CREATURE, [], effects)
    dm.CardRegistry.register_card_data(c2)
    
    # Add cards to hand
    # 2 High Value Cards, 1 Low Value Card
    # AI should discard Low Value Card to keep High Value ones (assuming heuristic prefers high power/cost/value)
    
    # Define a heuristic callback
    def heuristic_callback(states):
        # Determine value of state based on hand content for active player
        values = []
        for s in states:
            if not s:
                values.append(0.0)
                continue
            
            p = s.players[s.active_player_id] # Or check specifically player 0 if fixed
            hand_value = 0
            for c in p.hand:
                if c.card_id == 1000:
                    hand_value += 10
                elif c.card_id == 1001:
                    hand_value += 1
            
            # Normalize or just return raw score if MCTS uses value for comparison
            # MCTS usually expects 0-1 or -1 to 1.
            # Max possible is 30 (3 high value). Current is 21 (2 high, 1 low).
            # If we discard low, we have 20. If we discard high, we have 11.
            # So 20 > 11. 
            # We return value between 0 and 1.
            val = min(1.0, hand_value / 30.0)
            values.append(val)
        return values

    # Set callback
    # Note: bind_ai.cpp exposes set_batch_callback or set_flat_batch_callback?
    # batch_evaluator.cpp calls dm::python::has_batch_callback()
    # It likely expects `set_batch_callback` taking list of states.
    dm.set_batch_callback(heuristic_callback)

    try:
        iid = 100
        p0 = state.players[0]
        p0.hand.clear()
        
        # Add 2 High, 1 Low
        # Hand: [High(iid+1), High(iid+2), Low(iid+3)]
        # Indices: 0, 1, 2
        
        c_high1 = dm.CardStub(1000, iid+1)
        c_high2 = dm.CardStub(1000, iid+2)
        c_low = dm.CardStub(1001, iid+3)
        
        p0.hand.append(c_high1)
        p0.hand.append(c_high2)
        p0.hand.append(c_low)
        
        # Verify
        assert len(p0.hand) == 3
        
        # Create DISCARD command
        cmd = dm.CommandDef()
        cmd.type = dm.CommandType.DISCARD
        cmd.amount = 1
        cmd.target_group = 0 # Player 0
        
        print("Executing DISCARD command with AI...")
        
        # CommandSystemWrapper helper
        ctx = {}
        dm.CommandSystem.execute_command(state, cmd, -1, 0, ctx)
        
        print("Command executed.")
        
        # Check result
        # One card should be in graveyard
        assert len(p0.hand) == 2
        assert len(p0.graveyard) == 1
        
        discarded = p0.graveyard[0]
        print(f"AI Discarded Card ID: {discarded.card_id}")
        
        # Expect Low Value Card (1001) to be discarded
        assert discarded.card_id == 1001
        
    finally:
        dm.clear_batch_callback()

if __name__ == "__main__":
    test_ai_selection()
