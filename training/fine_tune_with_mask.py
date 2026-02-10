#!/usr/bin/env python3
"""Fine-tune policy head using synthetic playable/attack samples and legal-action masking.

Creates synthetic playable/attack GameState examples, computes legal masks via EngineCompat
and `dm_ai_module.CommandEncoder.command_to_index`, and fine-tunes policy head with a
masked-cross-entropy loss that ignores illegal actions.
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import StateTokenizer
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit import commands_v2 as commands
from dm_toolkit.training.command_compat import generate_legal_commands, normalize_to_command, command_to_index
import dm_ai_module
from dm_toolkit.action_to_command import map_action


def make_synthetic_examples(n_each: int = 100):
    import tools.emit_play_attack_states as emit
    tokenizer = StateTokenizer(max_len=200)
    card_db = {42: {'cost': 1}, 1001: {'cost': 0}, 2001: {'cost': 0}}

    states = []
    masks = []
    targets = []

    for _ in range(n_each):
        s = emit.make_playable_state()
        toks = tokenizer.encode_state(s, 0)
        # legal indices (prefer command-first via central helper)
        cmds = generate_legal_commands(s, card_db, strict=False) or []
        legal = [False] * dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE
        chosen = None
        from dm_toolkit.action_to_command import map_action
        from dm_toolkit.training.command_compat import normalize_to_command
        def _normalize_to_command(obj):
            try:
                if isinstance(obj, dict):
                    return obj
                if hasattr(obj, 'to_dict'):
                    try:
                        return obj.to_dict()
                    except Exception:
                        pass
                try:
                    return normalize_to_command(obj)
                except Exception:
                    return {'_repr': repr(obj)}
            except Exception:
                return {'_repr': repr(obj)}

        for c in cmds:
            d = _normalize_to_command(c)
            idx = dm_ai_module.CommandEncoder.command_to_index(d)
            if idx is not None and 0 <= idx < len(legal):
                legal[idx] = True
                # pick PLAY if present
                if chosen is None and str(d.get('type','')).upper() in ('PLAY_FROM_ZONE','PLAY','PLAY_CARD'):
                    chosen = idx
        # fallback choose argmax legal
        if chosen is None:
            for i,v in enumerate(legal):
                if v:
                    chosen = i; break
        if chosen is None:
            chosen = 0

        states.append(toks)
        masks.append(legal)
        targets.append(chosen)

    for _ in range(n_each):
        s = emit.make_attack_state()
        toks = tokenizer.encode_state(s, 0)
        cmds = generate_legal_commands(s, card_db, strict=False) or []
        legal = [False] * dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE
        chosen = None
        from dm_toolkit.action_to_command import map_action
        for c in cmds:
            d = _normalize_to_command(c)
            idx = dm_ai_module.CommandEncoder.command_to_index(d)
            if idx is not None and 0 <= idx < len(legal):
                legal[idx] = True
                if chosen is None and 'ATTACK' in str(d.get('type','')).upper():
                    chosen = idx
        if chosen is None:
            for i,v in enumerate(legal):
                if v:
                    chosen = i; break
        if chosen is None:
            chosen = 0

        states.append(toks)
        masks.append(legal)
        targets.append(chosen)

    return np.array(states, dtype=np.int64), np.array(masks, dtype=bool), np.array(targets, dtype=np.int64)


def load_base_dataset(npz_path: str):
    d = np.load(npz_path)
    states = d['states']
    policies = d['policies']
    targets = np.argmax(policies, axis=1).astype(np.int64)
    # default masks: all True (no masking)
    masks = np.ones((len(states), dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE), dtype=bool)
    return states, masks, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/transformer_training_data_converted.npz')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--out', type=str, default='models/checkpoints/policy_finetuned_masked.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Desired action dim
    ACTION_DIM = int(dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE)

    # Load base dataset
    base_states, base_masks, base_targets = load_base_dataset(args.data)

    # Create synthetic augmented examples
    syn_states, syn_masks, syn_targets = make_synthetic_examples(n_each=200)

    # Combine
    all_states = np.concatenate([base_states, syn_states], axis=0)
    all_masks = np.concatenate([base_masks, syn_masks], axis=0)
    all_targets = np.concatenate([base_targets, syn_targets], axis=0)

    ds = TensorDataset(torch.from_numpy(all_states).long(), torch.from_numpy(all_masks), torch.from_numpy(all_targets))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = DuelTransformer(vocab_size=1000, action_dim=ACTION_DIM, d_model=256, nhead=8, num_layers=6, max_len=200).to(device)
    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    model.load_state_dict(state, strict=False)

    # Freeze except policy head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.policy_head.parameters():
        p.requires_grad = True

    optim = torch.optim.AdamW([p for p in model.policy_head.parameters() if p.requires_grad], lr=5e-4)

    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for st_batch, mask_batch, targ_batch in dl:
            st_batch = st_batch.to(device)
            mask_batch = mask_batch.to(device)
            targ_batch = targ_batch.to(device)

            optim.zero_grad()
            # Clip token ids to model vocab size (export used vocab_size=1000)
            st_batch = (st_batch % 1000).to(st_batch.device)
            logits, _ = model(st_batch, padding_mask=(st_batch==0))

            # Apply mask: set illegal logits to large negative
            illegal = ~mask_batch
            logits_clone = logits.clone()
            logits_clone[illegal] = -1e9

            loss = nn.CrossEntropyLoss()(logits_clone, targ_batch)
            loss.backward()
            optim.step()
            tot += loss.item()

        print(f'Epoch {epoch} loss {tot/len(dl):.4f}')

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.out)
    print('Saved masked fine-tuned model to', args.out)


if __name__ == '__main__':
    main()
