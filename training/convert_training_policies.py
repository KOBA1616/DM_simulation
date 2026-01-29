#!/usr/bin/env python3
"""Convert existing transformer training data policy indices to current CommandEncoder indices.

Usage: python training/convert_training_policies.py
"""
import os
import numpy as np
from collections import Counter

project_root = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.insert(0, project_root)

import dm_ai_module


def main():
    src = os.path.join('data', 'transformer_training_data.npz')
    dst = os.path.join('data', 'transformer_training_data_converted.npz')
    if not os.path.exists(src):
        print('Source not found:', src)
        return

    d = np.load(src)
    states = d['states']
    policies = d['policies']
    values = d['values']
    n, old_dim = policies.shape

    if not hasattr(dm_ai_module, 'CommandEncoder') or dm_ai_module.CommandEncoder is None:
        print('CommandEncoder not available in dm_ai_module. Aborting.')
        return

    new_dim = dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE
    print('Converting policies: samples=', n, 'old_dim=', old_dim, 'new_dim=', new_dim)

    # Build mapping old_index -> new_index (or -1)
    mapping = [-1] * old_dim
    unresolved = []
    for i in range(old_dim):
        try:
            cmd_old = dm_ai_module.index_to_command(i, None)
            # Normalize the returned command into the shape expected by CommandEncoder
            t = cmd_old.get('type') if isinstance(cmd_old, dict) else None
            # Get type name if enum-like
            if hasattr(t, 'name'):
                tname = t.name
            else:
                tname = str(t)

            new_cmd = None
            if 'MANA' in tname:
                slot = int(cmd_old.get('slot_index', 0) or 0)
                # new encoder expects slot indices starting at MANA_CHARGE_BASE
                slot = slot + getattr(dm_ai_module.CommandEncoder, 'MANA_CHARGE_BASE', 1)
                new_cmd = {'type': 'MANA_CHARGE', 'slot_index': slot}
            elif 'PLAY' in tname:
                slot = int(cmd_old.get('slot_index', 0) or 0)
                new_cmd = {'type': 'PLAY_FROM_ZONE', 'slot_index': slot}
            elif 'PASS' in tname:
                new_cmd = {'type': 'PASS'}
            else:
                # best-effort map other types to PASS
                new_cmd = {'type': 'PASS'}

            new_i = dm_ai_module.CommandEncoder.command_to_index(new_cmd)
            mapping[i] = int(new_i)
        except Exception:
            mapping[i] = -1
            unresolved.append(i)

    print('Unresolved old indices count:', len(unresolved))

    new_policies = np.zeros((n, new_dim), dtype=np.float32)
    for s in range(n):
        for old_i in range(old_dim):
            prob = float(policies[s, old_i])
            ni = mapping[old_i]
            if ni >= 0 and ni < new_dim:
                new_policies[s, ni] += prob
            else:
                # fallback: add to PASS index 0
                new_policies[s, 0] += prob

    # normalize to probability distributions if rows sum > 0
    row_sums = new_policies.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    new_policies[mask] = new_policies[mask] / row_sums[mask]

    # diagnostics
    argmax = np.argmax(new_policies, axis=1)
    from collections import Counter
    cnt = Counter(argmax.tolist())
    print('Top new argmax:', cnt.most_common(10))

    # Save
    np.savez_compressed(dst, states=states, policies=new_policies, values=values)
    print('Saved converted dataset to', dst)


if __name__ == '__main__':
    main()
