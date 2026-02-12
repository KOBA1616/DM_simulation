import argparse
import logging
import logging.handlers
import random
import time
import traceback

LOG = 'tmp_selfplay_long.log'

# Configure logging as early as possible so downstream imports can't add
# their own handlers before central configuration. CLI will re-run
# `configure_logging` later with any overrides.
from scripts.logging_manager import configure_logging, get_logger
configure_logging(log_file=LOG)
import os
import glob

# If a native dm_ai_module is present in common build locations, prefer it
# by setting `DM_AI_MODULE_NATIVE` before importing toolkit code that
# ultimately imports `dm_ai_module`.
def _discover_native_pyd():
    root = os.path.dirname(os.path.dirname(__file__))
    candidates = []
    # common locations: bin/, build-msvc/, build-mingw/
    candidates += glob.glob(os.path.join(root, 'bin', 'dm_ai_module*.pyd'))
    candidates += glob.glob(os.path.join(root, 'build-msvc', '**', 'dm_ai_module*.pyd'), recursive=True)
    candidates += glob.glob(os.path.join(root, 'build-mingw', '**', 'dm_ai_module*.pyd'), recursive=True)
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None

native_p = _discover_native_pyd()
if native_p:
    os.environ.setdefault('DM_AI_MODULE_NATIVE', native_p)
    pyd_dir = os.path.dirname(native_p)
    cur_pp = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = pyd_dir + os.pathsep + cur_pp if cur_pp else pyd_dir

from dm_toolkit.gui.headless import create_session
from dm_toolkit import commands_v2

# Prefer command-first wrapper; provide safe fallback to legacy compatibility shim when necessary
def get_legal_commands(gs, card_db):
    try:
        cmds = commands_v2.generate_legal_commands(gs, card_db, strict=False) or []
    except Exception:
        try:
            cmds = commands_v2.generate_legal_commands(gs, card_db) or []
        except Exception:
            cmds = []
    return cmds

# module logger; use manager's `get_logger` so it inherits configured handlers
logger = get_logger('selfplay_long')

def run_one_game(max_steps=10000, max_actions_per_phase=50):
    sess = create_session()
    gs = sess.gs
    card_db = sess.card_db
    steps = 0
    exceptions = 0
    actions_executed = 0
    same_phase_count = 0
    # counts for command types we care about
    cmd_counts = {
        'MANA_CHARGE': 0,
        'PLAY_FROM_ZONE': 0,
        'ATTACK': 0,
    }
    cmd_samples = {k: [] for k in cmd_counts}
    while steps < max_steps and not sess.is_game_over():
        # Track phase stability to avoid getting stuck in a single phase
        cur_phase = getattr(gs, 'current_phase', None)
        if steps == 0:
            prev_phase = cur_phase
            same_phase_count = 0
        else:
            if cur_phase == prev_phase:
                same_phase_count += 1
            else:
                prev_phase = cur_phase
                same_phase_count = 0

        # Debug: log current phase and, if ATTACK, list battle-zone creatures and attributes
        try:
            ap = getattr(gs, 'active_player_id', getattr(gs, 'active_player', None))
            phase_name = getattr(cur_phase, 'name', None) or str(cur_phase)
            logger.debug(f"state_phase -> active_player={ap}, current_phase={phase_name}, raw={getattr(cur_phase,'value',cur_phase)}")
            if str(phase_name).upper() == 'ATTACK' or 'ATTACK' in str(phase_name).upper():
                players = getattr(gs, 'players', [])
                p = None
                try:
                    p = players[ap]
                except Exception:
                    p = players[0] if players else None
                battle = getattr(p, 'battle_zone', []) if p else []
                # Summarize attack-phase info to reduce log volume; keep small DEBUG sample
                logger.info("ATTACK_PHASE -> battle_count=%d", len(battle))
                if logger.isEnabledFor(logging.DEBUG):
                    creatures = []
                    for b in (battle or [])[:50]:
                        try:
                            info = {
                                'repr': repr(b),
                                'instance_id': getattr(b, 'instance_id', None),
                                'card_id': getattr(b, 'card_id', None),
                                'is_tapped': getattr(b, 'is_tapped', None),
                                'sick': getattr(b, 'sick', None),
                                'summoning_sickness': getattr(b, 'summoning_sickness', getattr(b, 'summoning_sick', None)),
                                'owner': getattr(b, 'owner', None),
                                'zone_index': getattr(b, 'zone_index', None),
                            }
                            # try can_attack method/property if present
                            try:
                                ca = getattr(b, 'can_attack', None)
                                if callable(ca):
                                    ca = ca()
                                info['can_attack'] = ca
                            except Exception:
                                info['can_attack'] = None
                            # try to extract a dict representation if available
                            try:
                                info['as_dict'] = b.to_dict()
                            except Exception:
                                info['as_dict'] = None
                            creatures.append(info)
                        except Exception:
                            logger.exception('Error while collecting creature info')
                    logger.debug('  BATTLE_ZONE_DUMP count=%d sample=%s', len(creatures), creatures[:10])
        except Exception:
            logger.exception('Error while logging phase/attack debug info')


            cmds = get_legal_commands(gs, card_db)
            if steps < 10 or steps % 100 == 0:
                types = []
                for c in cmds:
                    try:
                        types.append(c.to_dict().get('type', 'UNKNOWN'))
                    except:
                        types.append(str(c))
                logger.info(f"Step {steps}: Legal commands count={len(cmds)} types={types[:10]}")
        except Exception:
            logger.error("Failed to get legal commands for debug")


        # Use GameSession.step_game() to delegate logic to C++ engine
        # This handles action generation, selection, and execution internally
        try:
            sess.step_game()
            
            # After step, check what happened by looking at last action
            # Note: We can't easily count specific command types since step_game() encapsulates it
            # But we can check phase changes
            actions_executed += 1
            
        except Exception:
            exceptions += 1
            logger.exception('Error during sess.step_game()')

        # Continue loop until game over or max steps
        steps += 1


        # No need for manual execution loop since step_game() does it one step at a time


        steps += 1
    return {
        'steps': steps,
        'exceptions': exceptions,
        'actions': actions_executed,
        'game_over': sess.is_game_over(),
        'winner': getattr(sess.gs, 'winner', None) if sess.gs else None,
        'cmd_counts': cmd_counts,
        'cmd_samples': cmd_samples,
    }


def main(games=1, max_steps=10000, max_actions_per_phase=50):
    random.seed(42)
    results = []
    start = time.time()
    # Ensure single-game default; CLI may override but default remains 1
    for i in range(games):
        logger.info(f'Starting game {i}')
        try:
            r = run_one_game(max_steps=max_steps, max_actions_per_phase=max_actions_per_phase)
        except Exception:
            logger.exception(f'Unhandled exception running game {i}')
            r = {'steps': None, 'exceptions': None, 'actions': None, 'error': traceback.format_exc()}
        results.append(r)
        logger.info(f'Game {i}: {r}')
    duration = time.time() - start
    summary = {
        'games': games,
        'duration_s': duration,
        'results': results,
    }
    logger.info('SUMMARY:')
    logger.info(str(summary))
    logger.info('Done %s', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run headless self-play (long)')
    parser.add_argument('--games', type=int, default=1, help='Number of games to run (default 1)')
    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per game')
    parser.add_argument('--max-actions-per-phase', type=int, default=50, help='Max actions to attempt per phase')
    parser.add_argument('--log', type=str, default=LOG, help='Path to log file')
    parser.add_argument('--log-level', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Console log level (overrides DM_CONSOLE_LOG_LEVEL env)')
    parser.add_argument('--file-log-level', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='File handler log level (overrides DM_FILE_LOG_LEVEL env)')
    parser.add_argument('--root-log-level', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Root logger level (overrides DM_ROOT_LOG_LEVEL env)')
    parser.add_argument('--silent-loggers', type=str, default=None, help='Comma-separated logger names to silence (set to WARNING) - overrides DM_SILENT_LOGGERS env')
    args = parser.parse_args()
    # configure logging according to CLI / env
    silent = None
    if args.silent_loggers is not None:
        silent = [s.strip() for s in args.silent_loggers.split(',') if s.strip()]
    configure_logging(
        log_file=args.log,
        console_level_name=args.log_level,
        file_level_name=args.file_log_level,
        root_level_name=args.root_log_level,
        silent_loggers=silent,
    )
    # refresh module logger from centralized manager so it inherits handlers
    logger = get_logger('selfplay_long')
    # run games (default 1)
    main(games=args.games, max_steps=args.max_steps, max_actions_per_phase=args.max_actions_per_phase)
