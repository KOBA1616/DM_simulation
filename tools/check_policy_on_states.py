#!/usr/bin/env python3
"""Check model policy logits on playable and attack states.

Loads an ONNX DuelTransformer and runs inference on two contrived states
(playable, attack) produced by `tools/emit_play_attack_states.py`.

Prints:
- legal commands and their canonical indices
- model `policy_logits` top-k indices and scores
- where legal indices appear in model ranking
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.ai.agent.tokenization import StateTokenizer
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit import commands_v2 as commands
import dm_ai_module

try:
    import torch
except Exception:
    torch = None
try:
    import onnxruntime as ort
except Exception:
    ort = None


def find_onnx(path=None):
    if path and os.path.exists(path):
        return path
    mdl_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    mdl_dir = os.path.abspath(mdl_dir)
    if not os.path.isdir(mdl_dir):
        return None
    for f in os.listdir(mdl_dir):
        if f.endswith('.onnx'):
            return os.path.join(mdl_dir, f)
    return None


def load_session(onnx_path):
    if ort is None:
        raise RuntimeError('onnxruntime not available in environment')
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(onnx_path, so)


def model_policy_for_state(sess, tokenizer, state, player_id=0):
    tokens = tokenizer.encode_state(state, player_id)
    if isinstance(tokens, list):
        tokens = np.array(tokens, dtype=np.int64)
    else:
        tokens = tokens.astype(np.int64)
    # Ensure shape (1, max_len)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    # ONNX model exported with vocab_size=1000 in export script; clip tokens
    vocab_size = 1000
    tokens = np.mod(tokens, vocab_size).astype(np.int64)
    inp_name = 'input_ids'
    feed = {inp_name: tokens}
    out = sess.run(None, feed)
    policy_logits = out[0]
    return policy_logits


def legal_indices_for_state(state, card_db=None):
    card_db = card_db or {}
    try:
        try:
            cmds = commands.generate_legal_commands(state, card_db, strict=False) or []
        except TypeError:
            cmds = commands.generate_legal_commands(state, card_db) or []
        except Exception:
            cmds = []
    except Exception:
        cmds = []
    mapped = []
    for w in cmds:
        try:
            d = w.to_dict()
        except Exception:
            try:
                d = {'_repr': repr(w)}
            except Exception:
                d = {'_repr': str(w)}
        try:
            idx = dm_ai_module.CommandEncoder.command_to_index(d)
        except Exception:
            try:
                idx = dm_ai_module.CommandEncoder.command_to_index(w.to_dict() if hasattr(w, 'to_dict') else d)
            except Exception:
                idx = None
        mapped.append({'cmd': d, 'index': idx})
    return mapped


def is_pth(path: str) -> bool:
    return isinstance(path, str) and path.endswith('.pth')


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = find_onnx(model_path)
    if model_path is None:
        print('No model found in models/ and no path provided', file=sys.stderr)
        sys.exit(2)

    tokenizer = StateTokenizer(max_len=200)

    # Reuse creators from emit script by importing module
    import tools.emit_play_attack_states as emit

    # small card_db used by emit script
    card_db = {42: {'cost': 1}, 1001: {'cost': 0}, 2001: {'cost': 0}}

    states = [emit.make_playable_state(), emit.make_attack_state()]
    notes = ['playable_state', 'attack_state']

    is_pth_model = is_pth(model_path)
    sess = None
    model = None
    if is_pth_model:
        if torch is None:
            print('PyTorch not available to load .pth model', file=sys.stderr)
            sys.exit(2)
        print('Using PyTorch model:', model_path)
        ck = torch.load(model_path, map_location='cpu')
        state = ck.get('model_state_dict', ck)
        # infer desired action dim
        try:
            import dm_ai_module as _dm
            desired = int(_dm.CommandEncoder.TOTAL_COMMAND_SIZE)
        except Exception:
            desired = None
            for k, v in state.items():
                if k.endswith('policy_head.1.weight'):
                    desired = int(v.shape[0])
                    break
            if desired is None:
                print('Cannot infer action_dim from checkpoint', file=sys.stderr)
                sys.exit(2)
        from dm_toolkit.ai.agent.transformer_model import DuelTransformer
        model = DuelTransformer(vocab_size=1000, action_dim=desired, d_model=256, nhead=8, num_layers=6, max_len=200)
        model.load_state_dict(state, strict=False)
        model.eval()
    else:
        print('Using ONNX model:', model_path)
        sess = load_session(model_path)

    results = []
    for note, st in zip(notes, states):
        legal = legal_indices_for_state(st, card_db)
        if model is not None:
            # run PyTorch model
            import numpy as _np
            toks = tokenizer.encode_state(st, 0)
            toks = _np.array(toks, dtype=_np.int64)
            # Clip tokens to model vocab size used at export (1000)
            toks = np.mod(toks, 1000).astype(_np.int64)
            toks = toks.reshape(1, -1)
            inp = torch.from_numpy(toks)
            with torch.no_grad():
                logits, _ = model(inp)
            logits = logits.numpy()[0]
        else:
            policy = model_policy_for_state(sess, tokenizer, st, player_id=0)
            # policy shape (1, action_dim)
            logits = policy[0]
        # topk
        topk_idx = np.argsort(-logits)[:16]
        topk = [{'idx': int(int(i)), 'score': float(logits[int(i)])} for i in topk_idx]

        # Find each legal index rank
        ranks = []
        for m in legal:
            idx = m.get('index')
            rank = None
            score = None
            if idx is not None and 0 <= idx < logits.shape[0]:
                # rank is 0-based position in descending order
                rank = int(np.sum(logits > logits[idx]))
                score = float(logits[idx])
            ranks.append({'cmd': m.get('cmd'), 'index': idx, 'rank': rank, 'score': score})

        results.append({'note': note, 'model_action_dim': int(logits.shape[0]), 'topk': topk, 'legal': legal, 'legal_ranks': ranks})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
