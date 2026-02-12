
import os
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('verify_play')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_toolkit.gui.headless import create_session
from dm_toolkit import commands_v2

def run_verification(max_steps=200):
    logger.info("Starting Card Play Verification...")
    
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

    # Track battle zone counts
    battle_counts = {0: [], 1: []}
    
    # Run game steps
    steps = 0
    
    logger.info(f"{'Step':<5} | {'Turn':<5} | {'Phase':<15} | {'P0 Mana':<7} | {'P0 Battle':<9} | {'P1 Mana':<7} | {'P1 Battle':<9} | {'Action'}")
    logger.info("-" * 100)

    cards_played_count = 0

    try:
        while steps < max_steps and not sess.is_game_over():
            # Capture action if possible (by hooking or just observing state change)
            # headless.py or game_session.py doesn't easily expose the *last* action unless we hook callback
            
            last_action_str = ""
            def log_callback(msg):
                nonlocal last_action_str
                if "P0:" in msg or "P1:" in msg:
                    last_action_str = msg
            
            sess.callback_log = log_callback
            
            # Step the game
            sess.step_game()
            
            # Get current state info
            try:
                turn = getattr(gs, 'turn_number', steps)
                
                # Try to get phase name
                phase = getattr(gs, 'current_phase', 'UNKNOWN')
                phase_name = getattr(phase, 'name', str(phase))
                
                # Get counts
                p0_mana = len(getattr(gs.players[0], 'mana_zone', []))
                p0_battle = len(getattr(gs.players[0], 'battle_zone', []))
                
                p1_mana = len(getattr(gs.players[1], 'mana_zone', []))
                p1_battle = len(getattr(gs.players[1], 'battle_zone', []))
                
                battle_counts[0].append(p0_battle)
                battle_counts[1].append(p1_battle)
                
                if "PLAY" in last_action_str:
                    cards_played_count += 1
                    logger.info(f"{steps:<5} | {turn:<5} | {phase_name:<15} | {p0_mana:<7} | {p0_battle:<9} | {p1_mana:<7} | {p1_battle:<9} | {last_action_str}")
                elif steps % 10 == 0: # Log every 10 steps to keep it clean, unless action happened
                     logger.info(f"{steps:<5} | {turn:<5} | {phase_name:<15} | {p0_mana:<7} | {p0_battle:<9} | {p1_mana:<7} | {p1_battle:<9} | {last_action_str}")

                last_action_str = "" # Reset

            except Exception as e:
                logger.error(f"Error accessing state at step {steps}: {e}")
                
            steps += 1
            
    except Exception as e:
        logger.error(f"Error executing step {steps}: {e}")
        traceback.print_exc()

    logger.info("-" * 100)
    logger.info("Verification Complete.")
    
    # Analyze results
    p0_final_battle = len(getattr(gs.players[0], 'battle_zone', []))
    p1_final_battle = len(getattr(gs.players[1], 'battle_zone', []))
    
    logger.info(f"Final Battle Zone - P0: {p0_final_battle}, P1: {p1_final_battle}")
    logger.info(f"Total 'PLAY' actions detected: {cards_played_count}")
    
    if p0_final_battle > 0 or p1_final_battle > 0 or cards_played_count > 0:
        logger.info("SUCCESS: Cards were played.")
    else:
        logger.warning("FAILURE: No cards seem to have been played.")

if __name__ == "__main__":
    run_verification()
