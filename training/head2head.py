#!/usr/bin/env python3
"""Head-to-head evaluation between two ONNX models using native game loop."""
from pathlib import Path
import sys
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import onnxruntime as ort
import dm_ai_module as dm
import random
import time
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action

CARD_DB = dm.JsonLoader.load_cards('data/cards.json')


def make_session(onnx_path: str):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=['CPUExecutionProvider'])
    # Emit session input/output metadata for diagnostics
    try:
        inputs = []
        for inp in sess.get_inputs():
            inputs.append({'name': inp.name, 'shape': [d for d in inp.shape], 'type': inp.type})
        outputs = []
        for out in sess.get_outputs():
            outputs.append({'name': out.name, 'shape': [d for d in out.shape], 'type': out.type})
        info = {'event': 'sess_info', 'onnx': str(onnx_path), 'inputs': inputs, 'outputs': outputs}
        print("H2H_JSON: " + __import__('json').dumps(info, ensure_ascii=False))
    except Exception:
        pass
    # Warm-up run to avoid one-off startup cost during evaluation
    try:
        inp = sess.get_inputs()[0]
        shape = inp.shape
        # default length fallback
        seq_len = 200
        # if shape known and rank 2, use second dim when numeric
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            seq_len = shape[1]
        dummy = np.zeros((1, seq_len), dtype=np.int64)
        out_names = [o.name for o in sess.get_outputs()]
        sess.run(out_names, {inp.name: dummy})
    except Exception:
        pass
    return sess


def eval_policy_batch(sess, token_seqs):
    # token_seqs: list of sequences (each list[int] or numpy array)
    if len(token_seqs) == 0:
        return np.zeros((0, 0)), np.zeros((0,))
    arr = np.array(token_seqs, dtype=np.int64)
    inp = sess.get_inputs()[0].name
    # Diagnostic: print eval call shapes
    try:
        dbg = {'event': 'eval_call', 'input_name': inp, 'arr_shape': list(arr.shape)}
        print("H2H_JSON: " + __import__('json').dumps(dbg, ensure_ascii=False))
    except Exception:
        pass
    out_names = [o.name for o in sess.get_outputs()]
    transpose_used = False
    try:
        res = sess.run(out_names, {inp: arr})
    except Exception as e:
        # Try transpose fallback if the model expects seq-first input
        try:
            arr_t = arr.T
            dbg = {'event': 'eval_fallback', 'reason': str(e), 'tried_shape': list(arr_t.shape)}
            print("H2H_JSON: " + __import__('json').dumps(dbg, ensure_ascii=False))
            res = sess.run(out_names, {inp: arr_t})
            transpose_used = True
        except Exception:
            # re-raise original exception to be handled by caller
            raise
    policy = np.array(res[0])
    value = np.array(res[1]).squeeze(-1)
    # If the model returned seq-first outputs (seq_len, batch, ...), transpose to batch-first
    try:
        if policy.ndim == 3:
            # e.g., (seq_len, batch, action_dim) -> (batch, action_dim) by taking final token or aggregating
            # We assume model returns per-batch logits in second dimension; move batch to axis 0 and aggregate if needed.
            policy = policy.transpose(1, 0, 2)
            # If action dim present, reduce seq dimension by taking last token
            policy = policy[:, -1, :]
        elif policy.ndim == 2 and policy.shape[0] == arr.shape[1]:
            # policy shape (seq_len, action_dim) with seq_len == original batch -> transpose
            policy = policy.T
    except Exception:
        pass
    try:
        if value.ndim == 2 and value.shape[0] == arr.shape[1]:
            value = value.T.squeeze(-1)
    except Exception:
        pass
    return policy, value


def play_games_batch(sess_a, sess_b, seeds, max_steps=1000, progress_callback=None):
    # seeds: list of seeds for games to run in parallel
    n = len(seeds)
    instances = []
    for s in seeds:
        inst = dm.GameInstance(int(s), CARD_DB)
        # attempt canonical start_game, fall back to legacy starters and log failures
        started = False
        try:
            inst.start_game()
            started = True
        except Exception as e:
            try:
                dm.PhaseManager.start_game(inst.state, CARD_DB)
                started = True
            except Exception as e2:
                try:
                    inst.state.setup_test_duel()
                    started = True
                except Exception as e3:
                    try:
                        print("H2H_JSON: " + __import__('json').dumps({'event': 'start_game_failed', 'seed': int(s), 'errors': [str(e), str(e2), str(e3)]}, ensure_ascii=False))
                    except Exception:
                        pass
        # Emit diagnostics about which start method ran and basic zone counts
        try:
            p0 = EngineCompat.get_player(inst.state, 0)
            p1 = EngineCompat.get_player(inst.state, 1)
            start_diag = {'event': 'start_game_result', 'seed': int(s), 'started': bool(started), 'p0_counts': {'hand': len(getattr(p0, 'hand', [])), 'deck': len(getattr(p0, 'deck', [])), 'shields': len(getattr(p0, 'shield_zone', []))}, 'p1_counts': {'hand': len(getattr(p1, 'hand', [])), 'deck': len(getattr(p1, 'deck', [])), 'shields': len(getattr(p1, 'shield_zone', []))}}
            print("H2H_JSON: " + __import__('json').dumps(start_diag, ensure_ascii=False))
        except Exception:
            pass
        # If start_game returned but zones appear empty, try PhaseManager.start_game
        try:
            p0 = EngineCompat.get_player(inst.state, 0)
            p1 = EngineCompat.get_player(inst.state, 1)
            if started and (len(getattr(p0, 'deck', [])) == 0 and len(getattr(p1, 'deck', [])) == 0):
                try:
                    dm.PhaseManager.start_game(inst.state, CARD_DB)
                    p0_after = EngineCompat.get_player(inst.state, 0)
                    p1_after = EngineCompat.get_player(inst.state, 1)
                    pm_diag = {'event': 'start_game_phase_manager_applied', 'seed': int(s), 'p0_counts_after': {'hand': len(getattr(p0_after, 'hand', [])), 'deck': len(getattr(p0_after, 'deck', [])), 'shields': len(getattr(p0_after, 'shield_zone', []))}, 'p1_counts_after': {'hand': len(getattr(p1_after, 'hand', [])), 'deck': len(getattr(p1_after, 'deck', [])), 'shields': len(getattr(p1_after, 'shield_zone', []))}}
                    print("H2H_JSON: " + __import__('json').dumps(pm_diag, ensure_ascii=False))
                except Exception:
                    try:
                        print("H2H_JSON: " + __import__('json').dumps({'event': 'start_game_phase_manager_failed', 'seed': int(s)}, ensure_ascii=False))
                    except Exception:
                        pass
        except Exception:
            pass
        # Emit more verbose state debug to inspect why zones are empty
        try:
            try:
                state_repr = str(inst.state)
            except Exception:
                state_repr = None
            try:
                p0 = EngineCompat.get_player(inst.state, 0)
                p1 = EngineCompat.get_player(inst.state, 1)
                p0_attrs = [a for a in dir(p0) if not a.startswith('_')][:40]
                p1_attrs = [a for a in dir(p1) if not a.startswith('_')][:40]
            except Exception:
                p0 = p1 = None
                p0_attrs = p1_attrs = []
            st_dbg = {'event': 'state_debug', 'seed': int(s), 'state_repr': state_repr, 'p0_attrs_sample': p0_attrs, 'p1_attrs_sample': p1_attrs}
            print("H2H_JSON: " + __import__('json').dumps(st_dbg, ensure_ascii=False))
        except Exception:
            pass
        # If started but zones empty, attempt a forced test-duel setup as a recovery step
        try:
            p0 = EngineCompat.get_player(inst.state, 0)
            p1 = EngineCompat.get_player(inst.state, 1)
            if started and (len(getattr(p0, 'hand', [])) == 0 and len(getattr(p0, 'deck', [])) == 0 and len(getattr(p1, 'hand', [])) == 0 and len(getattr(p1, 'deck', [])) == 0):
                try:
                    inst.state.setup_test_duel()
                    p0b = EngineCompat.get_player(inst.state, 0)
                    p1b = EngineCompat.get_player(inst.state, 1)
                    forced = {'event': 'start_game_forced_setup', 'seed': int(s), 'p0_counts_after': {'hand': len(getattr(p0b, 'hand', [])), 'deck': len(getattr(p0b, 'deck', [])), 'shields': len(getattr(p0b, 'shield_zone', []))}, 'p1_counts_after': {'hand': len(getattr(p1b, 'hand', [])), 'deck': len(getattr(p1b, 'deck', [])), 'shields': len(getattr(p1b, 'shield_zone', []))}}
                    print("H2H_JSON: " + __import__('json').dumps(forced, ensure_ascii=False))
                except Exception:
                    try:
                        print("H2H_JSON: " + __import__('json').dumps({'event': 'start_game_forced_setup_failed', 'seed': int(s)}, ensure_ascii=False))
                    except Exception:
                        pass
                # If still empty, populate simple default deck and perform minimal setup (shields + hand)
                try:
                    p0c = EngineCompat.get_player(inst.state, 0)
                    p1c = EngineCompat.get_player(inst.state, 1)
                    if len(getattr(p0c, 'deck', [])) == 0 and len(getattr(p1c, 'deck', [])) == 0:
                        try:
                            default_deck = [1] * 40
                            try:
                                inst.state.set_deck(0, default_deck)
                                inst.state.set_deck(1, default_deck[:])
                            except Exception:
                                # Some bindings expect GameInstance API
                                try:
                                    inst.state.set_deck(0, default_deck)
                                except Exception:
                                    pass
                            # minimal manual populate: shields then hand
                            for pid in (0, 1):
                                p = EngineCompat.get_player(inst.state, pid)
                                try:
                                    for _ in range(5):
                                        if not getattr(p, 'deck', []): break
                                        p.shield_zone.append(p.deck.pop())
                                except Exception:
                                    pass
                                try:
                                    for _ in range(5):
                                        if not getattr(p, 'deck', []): break
                                        p.hand.append(p.deck.pop())
                                except Exception:
                                    pass
                            p0d = EngineCompat.get_player(inst.state, 0)
                            p1d = EngineCompat.get_player(inst.state, 1)
                            manual = {'event': 'start_game_manual_populated', 'seed': int(s), 'p0_counts': {'hand': len(getattr(p0d, 'hand', [])), 'deck': len(getattr(p0d, 'deck', [])), 'shields': len(getattr(p0d, 'shield_zone', []))}, 'p1_counts': {'hand': len(getattr(p1d, 'hand', [])), 'deck': len(getattr(p1d, 'deck', [])), 'shields': len(getattr(p1d, 'shield_zone', []))}}
                            print("H2H_JSON: " + __import__('json').dumps(manual, ensure_ascii=False))
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        # Emit initial state summary for debugging (hand/deck/shields counts)
        try:
            p0 = EngineCompat.get_player(inst.state, 0)
            p1 = EngineCompat.get_player(inst.state, 1)
            init = {'event': 'initial_state', 'seed': int(s), 'players': [
                {'hand': len(getattr(p0, 'hand', [])), 'deck': len(getattr(p0, 'deck', [])), 'shields': len(getattr(p0, 'shield_zone', []))},
                {'hand': len(getattr(p1, 'hand', [])), 'deck': len(getattr(p1, 'deck', [])), 'shields': len(getattr(p1, 'shield_zone', []))}
            ]}
            # Add small preview of contents for debugging empty zones
            try:
                def preview_zone(z):
                    try:
                        return [str(x) for x in list(z)[:5]] if z is not None else []
                    except Exception:
                        return []
                init['players_preview'] = [
                    {'hand_preview': preview_zone(getattr(p0, 'hand', None)), 'deck_preview': preview_zone(getattr(p0, 'deck', None)), 'shield_preview': preview_zone(getattr(p0, 'shield_zone', None))},
                    {'hand_preview': preview_zone(getattr(p1, 'hand', None)), 'deck_preview': preview_zone(getattr(p1, 'deck', None)), 'shield_preview': preview_zone(getattr(p1, 'shield_zone', None))}
                ]
            except Exception:
                pass
            print("H2H_JSON: " + __import__('json').dumps(init, ensure_ascii=False))
        except Exception:
            pass
        instances.append(inst)

    finished = [False] * n
    results = [0] * n
    # diagnostic storage per-instance
    last_seq = [None] * n
    last_policy_summary = [None] * n
    last_value = [None] * n
    last_chosen = [None] * n
    # track consecutive PASS-only choices to detect infinite/pass loops
    consecutive_pass_count = [0] * n
    PASS_LOOP_THRESHOLD = 8
    steps = 0
    last_emit = time.time()
    last_hb = last_emit
    while (not all(finished)) and steps < max_steps:
        # Collect required inferences for this step
        seqs_a = []
        idxs_a = []
        legals_a = []

        seqs_b = []
        idxs_b = []
        legals_b = []

        for i, inst in enumerate(instances):
            if finished[i]:
                continue
            state = inst.state
            if state.winner != dm.GameResult.NONE:
                finished[i] = True
                results[i] = 1 if state.winner == dm.GameResult.P1_WIN else (2 if state.winner == dm.GameResult.P2_WIN else 0)
                continue

            active = state.active_player_id
            legal = dm.IntentGenerator.generate_legal_actions(state, CARD_DB)
            if not legal:
                dm.PhaseManager.next_phase(state, CARD_DB)
                continue

            try:
                seq = dm.TensorConverter.convert_to_sequence(state, active, CARD_DB)
            except Exception:
                seq = [0] * 200

            # store last sequence for diagnostics
            try:
                last_seq[i] = list(seq) if hasattr(seq, '__iter__') else seq
            except Exception:
                last_seq[i] = None

            if active == 0:
                idxs_a.append(i)
                seqs_a.append(seq)
                legals_a.append(legal)
            else:
                idxs_b.append(i)
                seqs_b.append(seq)
                legals_b.append(legal)

        # Diagnostic: report how many sequences collected for each player
        try:
            dbg_pre = {'event': 'pre_eval', 'seqs_a': len(seqs_a), 'seqs_b': len(seqs_b)}
            print("H2H_JSON: " + __import__('json').dumps(dbg_pre, ensure_ascii=False))
        except Exception:
            pass

        # Run batched inference for player A and B
        try:
            if seqs_a:
                policies_a, vals_a = eval_policy_batch(sess_a, seqs_a)
            else:
                policies_a, vals_a = (np.zeros((0,)), np.zeros((0,)))
        except Exception as e:
            try:
                err = {'event': 'error', 'where': 'eval_policy_batch_a', 'message': str(e)}
                print("H2H_JSON: " + __import__('json').dumps(err, ensure_ascii=False))
            except Exception:
                pass
            # fallback: uniform random logits
            policies_a = np.ones((len(seqs_a), 600), dtype=np.float32) if seqs_a else np.zeros((0, 600))
            vals_a = np.zeros((len(seqs_a),), dtype=np.float32) if seqs_a else np.zeros((0,))

        try:
            if seqs_b:
                policies_b, vals_b = eval_policy_batch(sess_b, seqs_b)
            else:
                policies_b, vals_b = (np.zeros((0,)), np.zeros((0,)))
        except Exception as e:
            try:
                err = {'event': 'error', 'where': 'eval_policy_batch_b', 'message': str(e)}
                print("H2H_JSON: " + __import__('json').dumps(err, ensure_ascii=False))
            except Exception:
                pass
            policies_b = np.ones((len(seqs_b), 600), dtype=np.float32) if seqs_b else np.zeros((0, 600))
            vals_b = np.zeros((len(seqs_b),), dtype=np.float32) if seqs_b else np.zeros((0,))

        # Apply chosen actions for A
        for k, game_idx in enumerate(idxs_a):
            policy_logits = policies_a[k]
            # store top-5 policy summary and value for diagnostics
            try:
                top_idx = list(map(int, np.argsort(-policy_logits)[:5]))
                top_vals = [float(policy_logits[ii]) for ii in top_idx]
                last_policy_summary[game_idx] = list(zip(top_idx, top_vals))
            except Exception:
                last_policy_summary[game_idx] = None
            try:
                last_value[game_idx] = float(vals_a[k]) if len(vals_a) > k else None
            except Exception:
                last_value[game_idx] = None
            legal = legals_a[k]
            chosen = None
            # Emit raw legal actions and ActionEncoder mapping for diagnostics
            try:
                legal_map = []
                for a in legal:
                    try:
                        idx = dm.ActionEncoder.action_to_index(a)
                        legal_map.append({'repr': map_action(a) if True else str(a), 'index': int(idx) if idx is not None else None})
                    except Exception as _e:
                        try:
                            legal_map.append({'repr': map_action(a) if True else str(a), 'index': None, 'error': str(_e)})
                        except Exception:
                            legal_map.append({'repr': str(a), 'index': None, 'error': str(_e)})
                dbg_leg = {'event': 'legal_map', 'index': game_idx, 'turn': EngineCompat.get_turn_number(instances[game_idx].state), 'legal_map': legal_map}
                print("H2H_JSON: " + __import__('json').dumps(dbg_leg, ensure_ascii=False))
            except Exception:
                pass
            best_score = -1e9
            for act in legal:
                try:
                    idx = dm.ActionEncoder.action_to_index(act)
                except Exception:
                    idx = -1
                if idx is None or idx < 0 or idx >= len(policy_logits):
                    continue
                score = float(policy_logits[idx])
                if score > best_score:
                    best_score = score
                    chosen = act
            if chosen is None:
                chosen = random.choice(legal)
            # record chosen action representation
            try:
                # Prefer converting legacy Action objects to unified command dict
                last_chosen[game_idx] = map_action(chosen)
            except Exception:
                try:
                    if hasattr(chosen, 'to_dict'):
                        last_chosen[game_idx] = chosen.to_dict()
                    else:
                        last_chosen[game_idx] = str(chosen)
                except Exception:
                    last_chosen[game_idx] = None
            try:
                instances[game_idx].resolve_action(chosen)
            except Exception:
                try:
                    dm.GameLogicSystem.resolve_action(instances[game_idx].state, chosen, CARD_DB)
                except Exception:
                    pass
            try:
                if chosen.type == dm.ActionType.PASS:
                    dm.PhaseManager.next_phase(instances[game_idx].state, CARD_DB)
            except Exception:
                pass
            # update consecutive PASS detector and possibly force-draw
            try:
                is_pass = False
                try:
                    is_pass = ( (isinstance(last_chosen[game_idx], dict) and last_chosen[game_idx].get('type') == 'PASS') or getattr(chosen, 'type', None) == dm.ActionType.PASS )
                except Exception:
                    is_pass = getattr(chosen, 'type', None) == dm.ActionType.PASS
                if is_pass and len(legals_a[k]) == 1:
                    consecutive_pass_count[game_idx] += 1
                else:
                    consecutive_pass_count[game_idx] = 0
                if consecutive_pass_count[game_idx] >= PASS_LOOP_THRESHOLD:
                    finished[game_idx] = True
                    results[game_idx] = 0
                    try:
                        ev = {'event': 'forced_draw_due_to_pass_loop', 'index': game_idx, 'count': int(consecutive_pass_count[game_idx])}
                        print("H2H_JSON: " + __import__('json').dumps(ev, ensure_ascii=False))
                    except Exception:
                        pass
            except Exception:
                pass

        # Apply chosen actions for B
        for k, game_idx in enumerate(idxs_b):
            policy_logits = policies_b[k]
            # store top-5 policy summary and value for diagnostics
            try:
                top_idx = list(map(int, np.argsort(-policy_logits)[:5]))
                top_vals = [float(policy_logits[ii]) for ii in top_idx]
                last_policy_summary[game_idx] = list(zip(top_idx, top_vals))
            except Exception:
                last_policy_summary[game_idx] = None
            try:
                last_value[game_idx] = float(vals_b[k]) if len(vals_b) > k else None
            except Exception:
                last_value[game_idx] = None
            legal = legals_b[k]
            chosen = None
            # Emit raw legal actions and ActionEncoder mapping for diagnostics (player B)
            try:
                legal_map = []
                for a in legal:
                    try:
                        idx = dm.ActionEncoder.action_to_index(a)
                        legal_map.append({'repr': map_action(a) if True else str(a), 'index': int(idx) if idx is not None else None})
                    except Exception as _e:
                        try:
                            legal_map.append({'repr': map_action(a) if True else str(a), 'index': None, 'error': str(_e)})
                        except Exception:
                            legal_map.append({'repr': str(a), 'index': None, 'error': str(_e)})
                dbg_leg = {'event': 'legal_map', 'index': game_idx, 'turn': EngineCompat.get_turn_number(instances[game_idx].state), 'legal_map': legal_map}
                print("H2H_JSON: " + __import__('json').dumps(dbg_leg, ensure_ascii=False))
            except Exception:
                pass
            best_score = -1e9
            for act in legal:
                try:
                    idx = dm.ActionEncoder.action_to_index(act)
                except Exception:
                    idx = -1
                if idx is None or idx < 0 or idx >= len(policy_logits):
                    continue
                score = float(policy_logits[idx])
                if score > best_score:
                    best_score = score
                    chosen = act
            if chosen is None:
                chosen = random.choice(legal)
            # record chosen action representation
            try:
                last_chosen[game_idx] = map_action(chosen)
            except Exception:
                try:
                    if hasattr(chosen, 'to_dict'):
                        last_chosen[game_idx] = chosen.to_dict()
                    else:
                        last_chosen[game_idx] = str(chosen)
                except Exception:
                    last_chosen[game_idx] = None
            try:
                instances[game_idx].resolve_action(chosen)
            except Exception:
                try:
                    dm.GameLogicSystem.resolve_action(instances[game_idx].state, chosen, CARD_DB)
                except Exception:
                    pass
            try:
                if chosen.type == dm.ActionType.PASS:
                    dm.PhaseManager.next_phase(instances[game_idx].state, CARD_DB)
            except Exception:
                pass
            # update consecutive PASS detector and possibly force-draw (for B side)
            try:
                is_pass = False
                try:
                    is_pass = ( (isinstance(last_chosen[game_idx], dict) and last_chosen[game_idx].get('type') == 'PASS') or getattr(chosen, 'type', None) == dm.ActionType.PASS )
                except Exception:
                    is_pass = getattr(chosen, 'type', None) == dm.ActionType.PASS
                if is_pass and len(legals_b[k]) == 1:
                    consecutive_pass_count[game_idx] += 1
                else:
                    consecutive_pass_count[game_idx] = 0
                if consecutive_pass_count[game_idx] >= PASS_LOOP_THRESHOLD:
                    finished[game_idx] = True
                    results[game_idx] = 0
                    try:
                        ev = {'event': 'forced_draw_due_to_pass_loop', 'index': game_idx, 'count': int(consecutive_pass_count[game_idx])}
                        print("H2H_JSON: " + __import__('json').dumps(ev, ensure_ascii=False))
                    except Exception:
                        pass
            except Exception:
                pass

        steps += 1

        # Time-based periodic progress callback (emit at most once per second)
        now = time.time()
        if progress_callback and (now - last_emit >= 1.0):
            p1 = sum(1 for inst in instances if inst.state.winner == dm.GameResult.P1_WIN)
            p2 = sum(1 for inst in instances if inst.state.winner == dm.GameResult.P2_WIN)
            done = sum(1 for inst in instances if inst.state.winner != dm.GameResult.NONE)
            try:
                progress_callback(done, n, p1, p2)
            except Exception:
                pass
            last_emit = now

        # Heartbeat JSON for GUI visibility (every ~2s)
        if now - last_hb >= 2.0:
            try:
                hb = {'event': 'heartbeat', 'step': steps, 'done': done, 'total': n}
                print("H2H_JSON: " + __import__('json').dumps(hb, ensure_ascii=False))
            except Exception:
                pass
            last_hb = now

    # finalize results
    # If we exited because of max_steps, emit diagnostics for unfinished games
    if (not all(finished)) and steps >= max_steps:
        diags = []
        for i, inst in enumerate(instances):
            if finished[i]:
                continue
            try:
                legal = dm.IntentGenerator.generate_legal_actions(inst.state, CARD_DB) or []
                pending = EngineCompat.get_pending_effects_info(inst.state)
                # turn number and simple player zone summaries
                turn = EngineCompat.get_turn_number(inst.state)
                p0 = EngineCompat.get_player(inst.state, 0)
                p1 = EngineCompat.get_player(inst.state, 1)
                def zone_summary(p):
                    try:
                        return {
                            'hand': len(getattr(p, 'hand', [])),
                            'deck': len(getattr(p, 'deck', [])),
                            'battle': len(getattr(p, 'battle_zone', [])),
                            'mana': len(getattr(p, 'mana_zone', [])),
                            'shields': len(getattr(p, 'shield_zone', [])),
                        }
                    except Exception:
                        return {}

                di = {'index': i, 'active': getattr(inst.state, 'active_player_id', None), 'winner': int(getattr(inst.state, 'winner', 0)), 'legal_count': len(legal), 'pending_effects': pending, 'turn': turn, 'players': [zone_summary(p0), zone_summary(p1)]}
                # map legal actions to serializable form (best-effort)
                try:
                    mapped_legals = []
                    for a in legal:
                        try:
                            mapped_legals.append(map_action(a))
                        except Exception:
                            try:
                                mapped_legals.append(a.to_dict())
                            except Exception:
                                mapped_legals.append(str(a))
                    di['legal_actions'] = mapped_legals
                except Exception:
                    pass
                # attach tail of command history for context
                try:
                    history = EngineCompat.get_command_history(inst.state) or []
                    hist_tail = history[-12:]
                    hist_mapped = []
                    for c in hist_tail:
                        try:
                            hist_mapped.append(map_action(c))
                        except Exception:
                            try:
                                hist_mapped.append(c.to_dict())
                            except Exception:
                                hist_mapped.append(str(c))
                    di['command_history_tail'] = hist_mapped
                    # compute consecutive PASS count at tail
                    consec_pass = 0
                    for cmd in reversed(hist_mapped):
                        try:
                            t = cmd.get('type') if isinstance(cmd, dict) else None
                            if t == 'PASS' or (isinstance(cmd, str) and 'PASS' in cmd):
                                consec_pass += 1
                            else:
                                break
                        except Exception:
                            break
                    di['consecutive_pass_tail'] = consec_pass
                except Exception:
                    pass
                # attach diagnostic traces if available
                if last_seq[i] is not None:
                    di['last_seq_len'] = len(last_seq[i]) if hasattr(last_seq[i], '__len__') else None
                if last_policy_summary[i] is not None:
                    di['last_policy_top'] = last_policy_summary[i]
                if last_value[i] is not None:
                    di['last_value'] = last_value[i]
                if last_chosen[i] is not None:
                    di['last_chosen'] = last_chosen[i]
                diags.append(di)
            except Exception:
                diags.append({'index': i, 'error': 'diag_failed'})
        try:
            print("H2H_JSON: " + __import__('json').dumps({'event': 'max_steps_reached', 'steps': int(steps), 'unfinished': diags}, ensure_ascii=False))
        except Exception:
            pass

    # If all games finished but all were draws, emit per-game detailed dump
    try:
        if all((r == 0) for r in results) and len(results) > 0:
            det = []
            for i, inst in enumerate(instances):
                try:
                    legal = dm.IntentGenerator.generate_legal_actions(inst.state, CARD_DB) or []
                    pending = EngineCompat.get_pending_effects_info(inst.state)
                    turn = EngineCompat.get_turn_number(inst.state)
                    p0 = EngineCompat.get_player(inst.state, 0)
                    p1 = EngineCompat.get_player(inst.state, 1)
                    def zone_summary(p):
                        try:
                            return {
                                'hand': len(getattr(p, 'hand', [])),
                                'deck': len(getattr(p, 'deck', [])),
                                'battle': len(getattr(p, 'battle_zone', [])),
                                'mana': len(getattr(p, 'mana_zone', [])),
                                'shields': len(getattr(p, 'shield_zone', [])),
                            }
                        except Exception:
                            return {}

                    di = {'index': i, 'active': getattr(inst.state, 'active_player_id', None), 'winner': int(getattr(inst.state, 'winner', 0)), 'legal_count': len(legal), 'pending_effects': pending, 'turn': turn, 'players': [zone_summary(p0), zone_summary(p1)]}
                    try:
                        mapped_legals = []
                        for a in legal:
                            try:
                                mapped_legals.append(map_action(a))
                            except Exception:
                                try:
                                    mapped_legals.append(a.to_dict())
                                except Exception:
                                    mapped_legals.append(str(a))
                        di['legal_actions'] = mapped_legals
                    except Exception:
                        pass
                    if last_seq[i] is not None:
                        di['last_seq_len'] = len(last_seq[i]) if hasattr(last_seq[i], '__len__') else None
                    if last_policy_summary[i] is not None:
                        di['last_policy_top'] = last_policy_summary[i]
                    if last_value[i] is not None:
                        di['last_value'] = last_value[i]
                    if last_chosen[i] is not None:
                        di['last_chosen'] = last_chosen[i]
                    try:
                        history = EngineCompat.get_command_history(inst.state) or []
                        hist_tail = history[-12:]
                        hist_mapped = []
                        for c in hist_tail:
                            try:
                                hist_mapped.append(map_action(c))
                            except Exception:
                                try:
                                    hist_mapped.append(c.to_dict())
                                except Exception:
                                    hist_mapped.append(str(c))
                        di['command_history_tail'] = hist_mapped
                    except Exception:
                        pass
                    det.append(di)
                except Exception:
                    det.append({'index': i, 'error': 'diag_failed'})
            print("H2H_JSON: " + __import__('json').dumps({'event': 'detailed_group_draw_dump', 'details': det}, ensure_ascii=False))
    except Exception:
        pass

    for i, inst in enumerate(instances):
        if inst.state.winner == dm.GameResult.P1_WIN:
            results[i] = 1
        elif inst.state.winner == dm.GameResult.P2_WIN:
            results[i] = 2
        else:
            results[i] = 0

    return results


def head2head(onnx_a: str, onnx_b: str, games: int = 20, parallel: int = 1):
    """Run head-to-head: onnx_a as player1, onnx_b as player2 using parallel batched games.
    Returns p1_win_rate and wins dict.
    """
    sess_a = make_session(onnx_a)
    sess_b = make_session(onnx_b)

    # Quick check whether batched inference of size>1 works; if not, fall back to sequential mode
    try:
        test_seqs = [[0]*200 for _ in range(min(2, max(1, parallel)))]
        try:
            _p, _v = eval_policy_batch(sess_a, test_seqs)
            _p, _v = eval_policy_batch(sess_b, test_seqs)
        except Exception as e:
            msg = {'event': 'batch_not_supported', 'message': 'ONNX model rejects batched input, falling back to parallel=1', 'error': str(e)}
            print("H2H_JSON: " + __import__('json').dumps(msg, ensure_ascii=False))
            parallel = 1
    except Exception:
        # be permissive; if check fails, continue with given parallel
        pass

    wins = {0:0,1:0,2:0}

    # run games in chunks of `parallel`
    seeds = [int(time.time()*1000)%1000000 + i for i in range(games)]
    for start in range(0, games, parallel):
        group = seeds[start:start+parallel]

        # announce group start for GUI
        try:
            msg = {'event': 'group_start', 'group_index': start // parallel, 'group_size': len(group)}
            print("H2H_JSON: " + __import__('json').dumps(msg, ensure_ascii=False))
        except Exception:
            pass

        def progress_cb(done, total, p1, p2):
            try:
                draws = done - p1 - p2
                summary = {
                    'event': 'progress',
                    'group_done': done,
                    'group_total': total,
                    'wins': int(p1),
                    'losses': int(p2),
                    'draws': int(draws)
                }
                # Print one-line JSON for robust parsing by GUI
                print("H2H_JSON: " + __import__('json').dumps(summary, ensure_ascii=False))
            except Exception:
                pass

        results = play_games_batch(sess_a, sess_b, group, progress_callback=progress_cb)

        for r in results:
            wins[r] = wins.get(r, 0) + 1

        # periodic overall progress
        played = min(start + len(group), games)
        p1_wins = wins.get(1, 0)
        try:
            summary = {
                'event': 'summary',
                'played': played,
                'total_games': games,
                'wins': int(wins.get(1, 0)),
                'losses': int(wins.get(2, 0)),
                'draws': int(wins.get(0, 0))
            }
            print("H2H_JSON: " + __import__('json').dumps(summary, ensure_ascii=False))
        except Exception:
            pass

    total = games
    p1_wins = wins.get(1, 0)
    return p1_wins / total, wins


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--parallel', type=int, default=1, help='Number of parallel games per batch')
    args = p.parse_args()
    r, wins = head2head(args.a, args.b, games=args.games, parallel=args.parallel)
    print('p1_win_rate', r, 'wins', wins)
