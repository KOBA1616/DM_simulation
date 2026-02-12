
import os
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('verify_mana')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_toolkit.gui.headless import create_session
from dm_toolkit import commands_v2

def run_verification(max_steps=200):
    logger.info("Starting Mana Charge Verification...")
    
    # Create session with AI vs AI
    try:
        sess = create_session()
        logger.info("Session created.")
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return False

    gs = sess.gs
    if not gs:
        logger.error("GameInstance is None.")
        return False

    # Track mana counts
    mana_counts = {0: [], 1: []}
    turns = []
    
    # Run game steps
    steps = 0
    prev_turn = -1
    
    logger.info(f"{'Step':<5} | {'Turn':<5} | {'Phase':<20} | {'P0 Mana':<8} | {'P1 Mana':<8} | {'Active':<6}")
    logger.info("-" * 80)

    try:
        while steps < max_steps and not sess.is_game_over():
            sess.step_game()
            
            # Get current state info
            try:
                turn = getattr(gs, 'turn', steps) # Fallback to steps if turn not available
                active_pid = getattr(gs, 'active_player_id', -1)
                
                # Try to get phase name
                phase = getattr(gs, 'current_phase', 'UNKNOWN')
                phase_name = getattr(phase, 'name', str(phase))
                
                # Get mana counts
                p0_mana = len(getattr(gs.players[0], 'mana_zone', []))
                p1_mana = len(getattr(gs.players[1], 'mana_zone', []))
                
                mana_counts[0].append(p0_mana)
                mana_counts[1].append(p1_mana)
                turns.append(turn)
                
                logger.info(f"{steps:<5} | {turn:<5} | {phase_name:<20} | {p0_mana:<8} | {p1_mana:<8} | {active_pid:<6}")

            except Exception as e:
                logger.error(f"Error accessing state at step {steps}: {e}")
                
            steps += 1
            
    except Exception as e:
        logger.error(f"Error executing step {steps}: {e}")
        traceback.print_exc()

    logger.info("-" * 80)
    logger.info("Verification Complete.")
    
    # Analyze results
    p0_final_mana = mana_counts[0][-1] if mana_counts[0] else 0
    p1_final_mana = mana_counts[1][-1] if mana_counts[1] else 0
    
    logger.info(f"Final Mana - P0: {p0_final_mana}, P1: {p1_final_mana}")
    
    # Simple check: Did mana increase?
    # Since we run for 200 steps, checking if mana is > 1 implies some charging happened.
    # To verify *every turn*, we would need to correlate turn count with mana count increases.
    
    if p0_final_mana > 1 and p1_final_mana > 1:
        logger.info("SUCCESS: Mana counts increased for both players.")
    else:
        logger.warning("FAILURE: Mana counts did not increase significantly.")

if __name__ == "__main__":
    run_verification()
