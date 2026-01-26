#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
import json
import time
import logging
from typing import Any, List, Optional

# Lazy imports handled inside functions to avoid startup cost

def setup_logger(level_name: str):
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='%(levelname)s: %(message)s')
    return logging.getLogger('dm_toolkit.cli')

# --- Console (REPL) Logic ---

def show_status(sess: Any):
    gs = getattr(sess, 'gs', None)
    active = getattr(gs, 'active_player_id', None) if gs else None
    phase = getattr(gs, 'phase', None) or getattr(gs, 'current_phase', None) if gs else None
    print(f"Session: gs_present={'yes' if gs else 'no'} active_player={active} phase={phase}")

def list_hand(sess: Any, pid: int = 0):
    gs = getattr(sess, 'gs', None)
    if not gs:
        print("no game state")
        return
    try:
        p = gs.players[pid]
    except IndexError:
        print(f"no player {pid}")
        return
    hand = getattr(p, 'hand', []) or []
    if not hand:
        print("hand empty")
        return
    for c in hand:
        iid = getattr(c, 'instance_id', None)
        cid = getattr(c, 'card_id', None)
        print(f"iid={iid} card_id={cid} repr={c}")

def list_legal(sess: Any):
    if not getattr(sess, 'gs', None):
        print("no game state")
        return
    from dm_toolkit.commands import generate_legal_commands
    try:
        cmds = generate_legal_commands(sess.gs, sess.card_db)
    except Exception as e:
        print("generate_legal_commands failed:", e)
        return
    for i, c in enumerate(cmds):
        try:
            d = c.to_dict()
        except Exception:
            d = {'type': str(type(c))}
        print(f"[{i}] {d}")
    print(f"total legal commands: {len(cmds)}")

def play_first(sess: Any):
    from dm_toolkit.gui import headless
    gs = getattr(sess, 'gs', None)
    if not gs:
        return False
    p = gs.players[gs.active_player_id]
    hand = getattr(p, 'hand', []) or []
    for c in hand:
        iid = getattr(c, 'instance_id', None)
        if iid is not None:
            if headless.play_instance(sess, iid):
                print(f"played instance {iid}")
                return True
    return False

def run_console(args):
    from dm_toolkit.gui import headless
    logger = setup_logger(args.log_level)

    sess = headless.create_session(p0_human=args.p0_human, p1_human=args.p1_human)
    logger.info('Session created. gs present: %s', bool(getattr(sess, 'gs', None)))

    # REPL Loop
    print("Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            cmd_line = input('> ').strip()
        except EOFError:
            break
        if not cmd_line:
            continue

        parts = cmd_line.split()
        key = parts[0]

        if key in ('exit', 'quit'):
            break
        elif key == 'help':
            print("Commands: status, hand [pid], legal, step, play <iid>, play-first, run <steps>, dump")
        elif key == 'status':
            show_status(sess)
        elif key == 'hand':
            pid = int(parts[1]) if len(parts) > 1 else 0
            list_hand(sess, pid)
        elif key == 'legal':
            list_legal(sess)
        elif key == 'step':
            sess.step_phase()
            print("Stepped phase.")
        elif key == 'play':
            if len(parts) > 1:
                try:
                    iid = int(parts[1])
                    headless.play_instance(sess, iid)
                    print(f"Attempted play on {iid}")
                except ValueError:
                    print("Invalid ID")
            else:
                print("Usage: play <instance_id>")
        elif key == 'play-first':
            if not play_first(sess):
                print("No playable card found in hand.")
        elif key == 'run':
            n = int(parts[1]) if len(parts) > 1 else 10
            steps, over = headless.run_steps(sess, n)
            print(f"Run {steps} steps. Game Over: {over}")
        elif key == 'dump':
             gs = getattr(sess, 'gs', None)
             if gs:
                 for i, p in enumerate(gs.players):
                     print(f"Player {i}: Hand={len(p.hand)} Mana={len(p.mana_zone)} Shields={len(p.shield_zone)}")

# --- Simulation Logic ---

def run_sim(args):
    import random
    import builtins
    from dm_toolkit.training.evolution_ecosystem import EvolutionEcosystem

    # Filter noise
    if args.quiet:
        orig_print = builtins.print
        def filtered_print(*pargs, **kwargs):
            if pargs and isinstance(pargs[0], str) and ('EngineCompat' in pargs[0] or 'ExecuteCommand' in pargs[0]):
                return
            return orig_print(*pargs, **kwargs)
        builtins.print = filtered_print

    eco = EvolutionEcosystem(args.cards, args.meta)
    valid_ids = list(eco.card_db.keys())
    if not valid_ids:
        print("No cards loaded.")
        return

    # Deterministic seed
    if args.seed is not None:
        random.seed(args.seed)

    deck = [random.choice(valid_ids) for _ in range(40)]
    print(f"Running simulation with {args.games} games...")

    # 1. Quick stats
    stats = eco.collect_smart_stats(deck, deck, num_games=2)

    # 2. Full run
    eco.meta_decks = [{"name": "self", "cards": deck}]
    wr, score = eco.evaluate_deck(deck, "self", num_games=args.games)

    result = {
        "win_rate": wr,
        "score": score,
        "games": args.games,
        "top_played": sorted(((cid, v['play']) for cid, v in stats.items()), key=lambda x: -x[1])[:5]
    }

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.json}")
    else:
        print(f"Win Rate: {wr*100:.1f}% Score: {score:.2f}")

# --- Validation Logic ---

def run_validate(args):
    from dm_toolkit.gui.editor.data_manager import CardDataManager
    from dm_toolkit.editor.core.headless_impl import HeadlessEditorModel

    target_path = args.path
    print(f"Validating cards in {target_path}...")

    model = HeadlessEditorModel()
    manager = CardDataManager(model)

    if not os.path.exists(target_path):
        print(f"Path {target_path} not found.")
        sys.exit(1)

    errors = []

    if os.path.isdir(target_path):
        files = [f for f in os.listdir(target_path) if f.endswith('.json')]
        for fname in files:
            fpath = os.path.join(target_path, fname)
            _validate_file(manager, fpath, fname, errors)
    else:
        _validate_file(manager, target_path, os.path.basename(target_path), errors)

    if errors:
        print(f"Found {len(errors)} errors:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    else:
        print(f"Validation passed.")

def _validate_file(manager, path, fname, errors):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                _validate_single_card(manager, item, fname, errors)
        else:
             _validate_single_card(manager, data, fname, errors)
    except json.JSONDecodeError:
        errors.append(f"{fname}: Invalid JSON")
    except Exception as e:
        errors.append(f"{fname}: Unexpected error {e}")

def _validate_single_card(manager, data, fname, errors):
    # Minimal validation checks
    required = ['id', 'name', 'civilization', 'type', 'cost']
    missing = [f for f in required if f not in data]
    if missing:
        errors.append(f"{fname} (id={data.get('id')}): Missing fields {missing}")
        return

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(prog="dm-cli", description="Duel Masters Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Console
    p_console = subparsers.add_parser("console", help="Interactive headless console")
    p_console.add_argument("--p0-human", action="store_true")
    p_console.add_argument("--p1-human", action="store_true")
    p_console.add_argument("--log-level", default="INFO")
    p_console.set_defaults(func=run_console)

    # Sim
    p_sim = subparsers.add_parser("sim", help="Run headless simulation")
    p_sim.add_argument("--cards", default=os.path.join("data", "cards.json"))
    p_sim.add_argument("--meta", default=os.path.join("data", "meta_decks.json"))
    p_sim.add_argument("--games", type=int, default=100)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.add_argument("--quiet", action="store_true")
    p_sim.add_argument("--json", help="Output results to JSON file")
    p_sim.set_defaults(func=run_sim)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate card data")
    p_val.add_argument("--path", default=os.path.join("data", "cards.json"), help="Path to cards.json or directory of card JSONs")
    p_val.set_defaults(func=run_validate)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
