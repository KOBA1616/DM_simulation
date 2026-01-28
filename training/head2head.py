#!/usr/bin/env python3
"""Head-to-head evaluation between two ONNX models using native game loop."""
from pathlib import Path
import sys
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import onnxruntime as ort
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
import os
import glob

# Prefer native dm_ai_module (.pyd) when available so all matches use the C++ engine.
# If `DM_AI_MODULE_NATIVE` is set, prefer that path. Otherwise search common build output
# directories (bin/Release, build-msvc, build-mingw) for a dm_ai_module*.pyd and
# place its directory at the front of sys.path before importing.
native_override = os.environ.get('DM_AI_MODULE_NATIVE')
if native_override and os.path.exists(native_override):
    pyd_dir = os.path.dirname(native_override)
    if pyd_dir not in sys.path:
        sys.path.insert(0, pyd_dir)
else:
    # search common locations under repo_root
    candidates = []
    candidates.append(str(repo_root / 'bin' / 'Release'))
    candidates.append(str(repo_root / 'build-msvc'))
    candidates.append(str(repo_root / 'build-mingw'))
    for d in candidates:
        try:
            pattern = os.path.join(d, 'dm_ai_module*.pyd')
            found = glob.glob(pattern)
            if found:
                pyd_dir = os.path.dirname(found[0])
                if pyd_dir not in sys.path:
                    sys.path.insert(0, pyd_dir)
                # set env var for downstream tools/logs
                os.environ['DM_AI_MODULE_NATIVE'] = found[0]
                break
        except Exception:
            pass

import dm_ai_module as dm
import random
import time
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.action_to_command import map_action
from dm_toolkit import commands

CARD_DB = dm.JsonLoader.load_cards('data/cards.json')


def make_session(model_path: str, use_pytorch: bool = False):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_pytorch:
        # Load PyTorch DuelTransformer from checkpoint
        try:
            ckpt = torch.load(str(model_path), map_location='cpu')
            # Model hyperparams must match training defaults
            model = DuelTransformer(vocab_size=1000, action_dim=600, d_model=256, nhead=8, num_layers=6, max_len=200, synergy_matrix_path=None)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)
            model.eval()
            try:
                info = {'event': 'sess_info', 'pytorch_model': str(model_path)}
                print("H2H_JSON: " + __import__('json').dumps(info, ensure_ascii=False))
            except Exception:
                pass
            return model
        except Exception as e:
            raise
    else:
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=['CPUExecutionProvider'])
    # Emit session input/output metadata for diagnostics
    try:
        inputs = []
        for inp in sess.get_inputs():
            inputs.append({'name': inp.name, 'shape': [d for d in inp.shape], 'type': inp.type})
        outputs = []
        for out in sess.get_outputs():
            outputs.append({'name': out.name, 'shape': [d for d in out.shape], 'type': out.type})
        info = {'event': 'sess_info', 'onnx': str(model_path), 'inputs': inputs, 'outputs': outputs}
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
    # Dispatch between ONNX session and PyTorch model
    if isinstance(sess, torch.nn.Module):
        return eval_policy_batch_torch(sess, token_seqs)

    arr = np.array(token_seqs, dtype=np.int64)
    inp = sess.get_inputs()[0].name
    # Diagnostic: print eval call shapes
    try:
        dbg = {'event': 'eval_call', 'input_name': inp, 'arr_shape': list(arr.shape)}
        print("H2H_JSON: " + __import__('json').dumps(dbg, ensure_ascii=False))
        # also emit a small head of the input array for inspection
        try:
            # capture up to first 4 sequences and first 16 tokens
            r = int(min(4, arr.shape[0]))
            c = int(min(16, arr.shape[1])) if arr.ndim >= 2 else 0
            arr_head = [list(map(int, arr[i, :c])) for i in range(r)] if r > 0 else []
            dbg2 = {'event': 'eval_input_head', 'input_name': inp, 'rows': r, 'cols': c, 'arr_head': arr_head}
            print("H2H_JSON: " + __import__('json').dumps(dbg2, ensure_ascii=False))
        except Exception:
            pass
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


def eval_policy_batch_torch(model: torch.nn.Module, token_seqs):
    if len(token_seqs) == 0:
        return np.zeros((0, 0)), np.zeros((0,))
    # convert to tensor (batch, seq_len)
    arr = np.array(token_seqs, dtype=np.int64)
    t = torch.from_numpy(arr).long()
    with torch.no_grad():
        # create padding mask where 0 is PAD
        padding_mask = (t == 0)
        policy_logits, value = model(t, padding_mask=padding_mask)
        # policy_logits: (batch, action_dim) or (batch, seq_len, action_dim)
        if policy_logits.ndim == 3:
            policy_logits = policy_logits[:, -1, :]
        policy = policy_logits.cpu().numpy()
        value = value.squeeze(-1).cpu().numpy()
    return policy, value


def play_games_batch(sess_a, sess_b, seeds, max_steps=1000, progress_callback=None, pass_penalty: float = 0.0):
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

    # Emit GameResult enum debug so we can interpret winner values
    try:
        gr_none = getattr(dm, 'GameResult').NONE
        try:
            gr_none_val = int(gr_none)
        except Exception:
            gr_none_val = str(gr_none)
        gr_dbg = {'event': 'game_result_info', 'NONE': gr_none_val, 'type': type(gr_none).__name__}
        print("H2H_JSON: " + __import__('json').dumps(gr_dbg, ensure_ascii=False))
    except Exception:
        pass

    finished = [False] * n
    results = [0] * n
    # diagnostic storage per-instance
    last_seq = [None] * n
    last_policy_summary = [None] * n
    last_value = [None] * n
    last_chosen = [None] * n
    # statistics per-instance
    moves_count = [0] * n
    pass_count = [0] * n
    # action counts per turn per game: list of dict(turn -> dict(action_type -> count))
    action_by_turn = [dict() for _ in range(n)]
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
            # Loop-level heartbeat so we can see iteration/finalization behavior
            try:
                loop_dbg = {'event': 'loop_iter', 'index': i, 'finished': bool(finished[i])}
                print("H2H_JSON: " + __import__('json').dumps(loop_dbg, ensure_ascii=False))
            except Exception:
                pass
            if finished[i]:
                continue
            state = inst.state
            try:
                pre_legal_dbg = {'event': 'before_legal', 'index': i, 'state_repr': str(state)[:120], 'winner': getattr(state, 'winner', None)}
                print("H2H_JSON: " + __import__('json').dumps(pre_legal_dbg, ensure_ascii=False))
            except Exception:
                pass
            try:
                w = getattr(state, 'winner', None)
                # Treat positive winner codes as terminal (1=P1 win, 2=P2 win). Some bindings use -1 for NONE.
                if w is not None and int(w) > 0:
                    finished[i] = True
                    results[i] = 1 if int(w) == int(getattr(dm.GameResult, 'P1_WIN', 1)) else (2 if int(w) == int(getattr(dm.GameResult, 'P2_WIN', 2)) else 0)
                    continue
            except Exception:
                pass

            active = state.active_player_id
            try:
                legal = dm.IntentGenerator.generate_legal_commands(state, CARD_DB) or []
            except Exception:
                legal = []
            if not legal:
                try:
                    legal = commands.generate_legal_commands(state, CARD_DB) or []
                except Exception:
                    legal = []

            # Emit minimal legal/active inspection so we always know why sequences may be skipped
            try:
                legal_inspect = {
                    'event': 'legal_inspect',
                    'index': i,
                    'active': int(active) if isinstance(active, (int,)) else str(active),
                    'legal_len': len(legal) if hasattr(legal, '__len__') else None,
                    'legal_type': type(legal).__name__
                }
                print("H2H_JSON: " + __import__('json').dumps(legal_inspect, ensure_ascii=False))
            except Exception:
                pass
            # Emit fuller debug: state repr and sample of legal actions (first 8)
            try:
                try:
                    state_repr = str(state)
                except Exception:
                    state_repr = None
                legal_sample = []
                try:
                    for a in list(legal)[:8]:
                        try:
                            # map_action may fail; prefer best-effort serializable form
                            legal_sample.append(map_action(a))
                        except Exception:
                            try:
                                legal_sample.append(a.to_dict() if hasattr(a, 'to_dict') else str(a))
                            except Exception:
                                legal_sample.append(str(a))
                except Exception:
                    legal_sample = None
                full_dbg = {'event': 'legal_full', 'index': i, 'state_repr': state_repr, 'legal_sample': legal_sample}
                print("H2H_JSON: " + __import__('json').dumps(full_dbg, ensure_ascii=False))
            except Exception:
                pass
            if not legal:
                dm.PhaseManager.next_phase(state, CARD_DB)
                continue

            try:
                seq = dm.TensorConverter.convert_to_sequence(state, active, CARD_DB)
            except Exception:
                seq = [0] * 200

            # Dump the raw sequence produced by TensorConverter for debugging
            try:
                seq_type = type(seq).__name__
                seq_len = len(seq) if hasattr(seq, '__len__') else None
                # capture a small head for inspection
                try:
                    seq_head = list(seq)[:16] if hasattr(seq, '__iter__') else None
                except Exception:
                    seq_head = None
                seq_dbg = {'event': 'seq_dump', 'index': i, 'seq_type': seq_type, 'seq_len': seq_len, 'seq_head': seq_head}
                print("H2H_JSON: " + __import__('json').dumps(seq_dbg, ensure_ascii=False))
            except Exception:
                pass

            # store last sequence for diagnostics
            try:
                last_seq[i] = list(seq) if hasattr(seq, '__iter__') else seq
            except Exception:
                last_seq[i] = None

            # Detailed per-instance diagnostic to help root-cause seqs==0
            try:
                inst_dbg = {
                    'event': 'instance_inspect',
                    'index': i,
                    'active': int(active) if isinstance(active, (int,)) else str(active),
                    'legal_count': len(legal) if hasattr(legal, '__len__') else (0 if not legal else 1),
                    'seq_type': type(seq).__name__,
                    'seq_len': len(seq) if hasattr(seq, '__len__') else None,
                }
                print("H2H_JSON: " + __import__('json').dumps(inst_dbg, ensure_ascii=False))
            except Exception:
                pass

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
                        idx = dm.CommandEncoder.command_to_index(map_action(a) if hasattr(a, 'to_dict') or isinstance(a, dict) else a if isinstance(a, dict) else (a.to_dict() if hasattr(a, 'to_dict') else a))
                        legal_map.append({'repr': map_action(a) if True else str(a), 'index': int(idx) if idx is not None else None})
                    except Exception as _e:
                        try:
                            legal_map.append({'repr': map_action(a) if True else str(a), 'index': None, 'error': str(_e)})
                        except Exception:
                            legal_map.append({'repr': str(a), 'index': None, 'error': str(_e)})
                # emit legal_map and index diagnostics
                try:
                    policy_len = len(policy_logits) if hasattr(policy_logits, '__len__') else None
                    # count invalid indices relative to policy length
                    invalid_count = 0
                    valid_count = 0
                    max_idx = -1
                    for lm in legal_map:
                        idx = lm.get('index')
                        if idx is None:
                            invalid_count += 1
                        else:
                            try:
                                if policy_len is not None and (int(idx) < 0 or int(idx) >= int(policy_len)):
                                    invalid_count += 1
                                else:
                                    valid_count += 1
                                if isinstance(idx, int) and idx > max_idx:
                                    max_idx = idx
                            except Exception:
                                invalid_count += 1
                    dbg_leg = {'event': 'legal_map', 'index': game_idx, 'turn': EngineCompat.get_turn_number(instances[game_idx].state), 'legal_map': legal_map, 'policy_len': policy_len, 'valid_indices': valid_count, 'invalid_indices': invalid_count, 'max_legal_index': max_idx}
                except Exception:
                    dbg_leg = {'event': 'legal_map', 'index': game_idx, 'turn': EngineCompat.get_turn_number(instances[game_idx].state), 'legal_map': legal_map}
                print("H2H_JSON: " + __import__('json').dumps(dbg_leg, ensure_ascii=False))
            except Exception:
                pass
            # Mask illegal indices (set to -inf) so selection only considers legal actions
            try:
                legal_indices = [lm.get('index') for lm in legal_map if lm.get('index') is not None]
                policy_masked = np.full_like(policy_logits, -np.inf, dtype=float)
                if legal_indices:
                    for li in legal_indices:
                        try:
                            if li is not None and 0 <= int(li) < len(policy_logits):
                                policy_masked[int(li)] = float(policy_logits[int(li)])
                        except Exception:
                            continue
                else:
                    # no legal indices, keep mask (will force random choice later)
                    policy_masked = np.full_like(policy_logits, -np.inf, dtype=float)
                # Emit top-k from masked logits
                try:
                    k = 5
                    topo = list(map(int, np.argsort(-policy_masked)[:k]))
                    topo_scores = [float(policy_masked[ii]) if not np.isneginf(policy_masked[ii]) else None for ii in topo]
                    topo_info = []
                    legal_set = set([int(x) for x in legal_indices]) if legal_indices else set()
                    for ii, sc in zip(topo, topo_scores):
                        topo_info.append({'index': int(ii), 'score': sc, 'is_legal': (int(ii) in legal_set)})
                    print("H2H_JSON: " + __import__('json').dumps({'event': 'policy_topk', 'index': game_idx, 'topk': topo_info}, ensure_ascii=False))
                except Exception:
                    pass
            except Exception:
                policy_masked = np.full_like(policy_logits, -np.inf, dtype=float)
            best_score = -1e9
            for act in legal:
                try:
                    # Prefer command-based encoding; accept legacy Action or dict-like
                    if isinstance(act, dict):
                        idx = dm.CommandEncoder.command_to_index(act)
                    elif hasattr(act, 'to_dict'):
                        idx = dm.CommandEncoder.command_to_index(act.to_dict())
                    else:
                        idx = dm.ActionEncoder.action_to_index(act)
                except Exception:
                    idx = -1
                if idx is None:
                    continue
                try:
                    if int(idx) < 0 or int(idx) >= len(policy_logits):
                        continue
                except Exception:
                    continue
                score = float(policy_masked[int(idx)]) if not np.isneginf(policy_masked[int(idx)]) else -np.inf
                # apply small penalty to PASS actions if configured
                try:
                    is_pass_act = dm.is_action_type(act, dm.ActionType.PASS)
                except Exception:
                    is_pass_act = False
                if is_pass_act and pass_penalty:
                    score -= float(pass_penalty)
                if score > best_score:
                    best_score = score
                    chosen = act
            if chosen is None:
                chosen = random.choice(legal)
            # compute chosen index for diagnostics
            try:
                try:
                    if isinstance(chosen, dict):
                        chosen_idx = dm.CommandEncoder.command_to_index(chosen)
                    elif hasattr(chosen, 'to_dict'):
                        chosen_idx = dm.CommandEncoder.command_to_index(chosen.to_dict())
                    else:
                        chosen_idx = dm.ActionEncoder.action_to_index(chosen)
                except Exception:
                    chosen_idx = None
                chosen_score = float(policy_masked[int(chosen_idx)]) if chosen_idx is not None and 0 <= int(chosen_idx) < len(policy_masked) and not np.isneginf(policy_masked[int(chosen_idx)]) else None
                print("H2H_JSON: " + __import__('json').dumps({'event': 'chosen_action', 'index': game_idx, 'chosen_index': (int(chosen_idx) if chosen_idx is not None else None), 'chosen_score': chosen_score, 'chosen_repr': map_action(chosen) if True else str(chosen)}, ensure_ascii=False))
            except Exception:
                pass
            else:
                # If model chose PASS but other non-PASS legal actions exist,
                # prefer a non-PASS action to avoid sterile PASS-only loops.
                try:
                    is_chosen_pass = False
                    try:
                        is_chosen_pass = dm.is_action_type(last_chosen[game_idx], dm.ActionType.PASS) or dm.is_action_type(chosen, dm.ActionType.PASS)
                    except Exception:
                        is_chosen_pass = dm.is_action_type(chosen, dm.ActionType.PASS)
                    if is_chosen_pass:
                        non_pass_candidates = [a for a in legal if not dm.is_action_type(a, dm.ActionType.PASS)]
                        if non_pass_candidates:
                            try:
                                print("H2H_JSON: " + __import__('json').dumps({'event': 'fallback_non_pass_selected', 'index': game_idx, 'reason': 'model_chose_pass_but_nonpass_available'}, ensure_ascii=False))
                            except Exception:
                                pass
                            chosen = random.choice(non_pass_candidates)
                except Exception:
                    pass
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
            # update stats
            try:
                moves_count[game_idx] += 1
                turn_no = EngineCompat.get_turn_number(instances[game_idx].state)
                act_type = None
                try:
                    if isinstance(last_chosen[game_idx], dict):
                        act_type = last_chosen[game_idx].get('type')
                except Exception:
                    pass
                if act_type is None:
                    try:
                        act_type = getattr(chosen, 'type', None)
                        if isinstance(act_type, int):
                            act_type = str(act_type)
                    except Exception:
                        act_type = 'UNKNOWN'
                if act_type == None:
                    act_type = 'UNKNOWN'
                # pass count
                if act_type == 'PASS' or dm.is_action_type(chosen, dm.ActionType.PASS):
                    pass_count[game_idx] += 1
                # turn distribution
                tb = action_by_turn[game_idx].setdefault(str(turn_no), {})
                tb[act_type] = tb.get(act_type, 0) + 1
            except Exception:
                pass
            try:
                from dm_toolkit.compat_wrappers import execute_action_compat
                execute_action_compat(instances[game_idx].state, chosen, CARD_DB)
            except Exception:
                try:
                    dm.GameLogicSystem.resolve_action(instances[game_idx].state, chosen, CARD_DB)
                except Exception:
                    pass
            try:
                if dm.is_action_type(chosen, dm.ActionType.PASS):
                    dm.PhaseManager.next_phase(instances[game_idx].state, CARD_DB)
            except Exception:
                pass
            # update consecutive PASS detector and possibly force-draw
            try:
                is_pass = False
                try:
                    is_pass = ( (isinstance(last_chosen[game_idx], dict) and last_chosen[game_idx].get('type') == 'PASS') or dm.is_action_type(chosen, dm.ActionType.PASS) )
                except Exception:
                    is_pass = dm.is_action_type(chosen, dm.ActionType.PASS)
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
                        if isinstance(a, dict):
                            idx = dm.CommandEncoder.command_to_index(a)
                        elif hasattr(a, 'to_dict'):
                            idx = dm.CommandEncoder.command_to_index(a.to_dict())
                        else:
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
            # Mask illegal indices (player B) and emit top-k from masked logits
            try:
                legal_indices = [lm.get('index') for lm in legal_map if lm.get('index') is not None]
                policy_masked = np.full_like(policy_logits, -np.inf, dtype=float)
                if legal_indices:
                    for li in legal_indices:
                        try:
                            if li is not None and 0 <= int(li) < len(policy_logits):
                                policy_masked[int(li)] = float(policy_logits[int(li)])
                        except Exception:
                            continue
                try:
                    k = 5
                    topo = list(map(int, np.argsort(-policy_masked)[:k]))
                    topo_scores = [float(policy_masked[ii]) if not np.isneginf(policy_masked[ii]) else None for ii in topo]
                    topo_info = []
                    legal_set = set([int(x) for x in legal_indices]) if legal_indices else set()
                    for ii, sc in zip(topo, topo_scores):
                        topo_info.append({'index': int(ii), 'score': sc, 'is_legal': (int(ii) in legal_set)})
                    print("H2H_JSON: " + __import__('json').dumps({'event': 'policy_topk', 'index': game_idx, 'topk': topo_info}, ensure_ascii=False))
                except Exception:
                    pass
            except Exception:
                policy_masked = np.full_like(policy_logits, -np.inf, dtype=float)
            best_score = -1e9
            for act in legal:
                try:
                    if isinstance(act, dict):
                        idx = dm.CommandEncoder.command_to_index(act)
                    elif hasattr(act, 'to_dict'):
                        idx = dm.CommandEncoder.command_to_index(act.to_dict())
                    else:
                        idx = dm.ActionEncoder.action_to_index(act)
                except Exception:
                    idx = -1
                if idx is None:
                    continue
                try:
                    if int(idx) < 0 or int(idx) >= len(policy_logits):
                        continue
                except Exception:
                    continue
                score = float(policy_masked[int(idx)]) if not np.isneginf(policy_masked[int(idx)]) else -np.inf
                # apply small penalty to PASS actions if configured
                try:
                    is_pass_act = dm.is_action_type(act, dm.ActionType.PASS)
                except Exception:
                    is_pass_act = False
                if is_pass_act and pass_penalty:
                    score -= float(pass_penalty)
                if score > best_score:
                    best_score = score
                    chosen = act
            if chosen is None:
                chosen = random.choice(legal)
            # compute chosen index for diagnostics (player B)
            try:
                try:
                    if isinstance(chosen, dict):
                        chosen_idx = dm.CommandEncoder.command_to_index(chosen)
                    elif hasattr(chosen, 'to_dict'):
                        chosen_idx = dm.CommandEncoder.command_to_index(chosen.to_dict())
                    else:
                        chosen_idx = dm.ActionEncoder.action_to_index(chosen)
                except Exception:
                    chosen_idx = None
                chosen_score = float(policy_masked[int(chosen_idx)]) if chosen_idx is not None and 0 <= int(chosen_idx) < len(policy_masked) and not np.isneginf(policy_masked[int(chosen_idx)]) else None
                print("H2H_JSON: " + __import__('json').dumps({'event': 'chosen_action', 'index': game_idx, 'chosen_index': (int(chosen_idx) if chosen_idx is not None else None), 'chosen_score': chosen_score, 'chosen_repr': map_action(chosen) if True else str(chosen)}, ensure_ascii=False))
            except Exception:
                pass
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
            # update stats for player B
            try:
                moves_count[game_idx] += 1
                turn_no = EngineCompat.get_turn_number(instances[game_idx].state)
                act_type = None
                try:
                    if isinstance(last_chosen[game_idx], dict):
                        act_type = last_chosen[game_idx].get('type')
                except Exception:
                    pass
                if act_type is None:
                    try:
                        act_type = getattr(chosen, 'type', None)
                        if isinstance(act_type, int):
                            act_type = str(act_type)
                    except Exception:
                        act_type = 'UNKNOWN'
                if act_type == None:
                    act_type = 'UNKNOWN'
                if act_type == 'PASS' or dm.is_action_type(chosen, dm.ActionType.PASS):
                    pass_count[game_idx] += 1
                tb = action_by_turn[game_idx].setdefault(str(turn_no), {})
                tb[act_type] = tb.get(act_type, 0) + 1
            except Exception:
                pass
            try:
                from dm_toolkit.compat_wrappers import execute_action_compat
                # Prefer instance-level API when available, but route through
                # compatibility helper to ensure command path execution.
                execute_action_compat(instances[game_idx].state, chosen, CARD_DB)
            except Exception:
                try:
                    dm.GameLogicSystem.resolve_action(instances[game_idx].state, chosen, CARD_DB)
                except Exception:
                    pass
            try:
                if dm.is_action_type(chosen, dm.ActionType.PASS):
                    dm.PhaseManager.next_phase(instances[game_idx].state, CARD_DB)
            except Exception:
                pass
            # update consecutive PASS detector and possibly force-draw (for B side)
            try:
                is_pass = False
                try:
                    is_pass = ( (isinstance(last_chosen[game_idx], dict) and last_chosen[game_idx].get('type') == 'PASS') or dm.is_action_type(chosen, dm.ActionType.PASS) )
                except Exception:
                    is_pass = dm.is_action_type(chosen, dm.ActionType.PASS)
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
                try:
                    legal = dm.IntentGenerator.generate_legal_commands(inst.state, CARD_DB) or []
                except Exception:
                    legal = []
                if not legal:
                    try:
                        legal = commands.generate_legal_commands(inst.state, CARD_DB) or []
                    except Exception:
                        legal = []
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
                    try:
                        legal = dm.IntentGenerator.generate_legal_commands(inst.state, CARD_DB) or []
                    except Exception:
                        legal = []
                    if not legal:
                        try:
                            legal = commands.generate_legal_commands(inst.state, CARD_DB) or []
                        except Exception:
                            legal = []
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

    # Aggregate statistics across games
    try:
        total_moves = sum(moves_count) if moves_count else 0
        total_passes = sum(pass_count) if pass_count else 0
        avg_game_length = float(total_moves) / len(moves_count) if len(moves_count) > 0 else 0.0
        pass_rate = float(total_passes) / total_moves if total_moves > 0 else 0.0
        # aggregate turn-action distribution
        agg_turn = {}
        for g in action_by_turn:
            for turn, acts in g.items():
                ta = agg_turn.setdefault(turn, {})
                for atype, cnt in acts.items():
                    ta[atype] = ta.get(atype, 0) + int(cnt)
        stats = {'event': 'h2h_stats', 'avg_game_length': avg_game_length, 'pass_rate': pass_rate, 'total_moves': int(total_moves), 'total_passes': int(total_passes), 'turn_action_distribution': agg_turn}
        try:
            print("H2H_JSON: " + __import__('json').dumps(stats, ensure_ascii=False))
        except Exception:
            pass
    except Exception:
        pass

    return results


def head2head(onnx_a: str, onnx_b: str, games: int = 20, parallel: int = 1, use_pytorch: bool = False, pass_penalty: float = 0.0):
    """Run head-to-head: onnx_a as player1, onnx_b as player2 using parallel batched games.
    Returns p1_win_rate and wins dict.
    """
    sess_a = make_session(onnx_a, use_pytorch=use_pytorch)
    sess_b = make_session(onnx_b, use_pytorch=use_pytorch)

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

        results = play_games_batch(sess_a, sess_b, group, progress_callback=progress_cb, pass_penalty=pass_penalty)

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
    p.add_argument('--pass-penalty', type=float, default=0.0, help='Small amount subtracted from PASS action logit to discourage PASS')
    p.add_argument('--use_pytorch', action='store_true', help='Load PyTorch .pth checkpoints instead of ONNX')
    args = p.parse_args()
    r, wins = head2head(args.a, args.b, games=args.games, parallel=args.parallel, use_pytorch=bool(getattr(args, 'use_pytorch', False)), pass_penalty=float(getattr(args, 'pass_penalty', 0.0)))
    print('p1_win_rate', r, 'wins', wins)
