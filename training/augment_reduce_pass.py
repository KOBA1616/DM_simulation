#!/usr/bin/env python3
"""Simple augmentation: reduce PASS probability mass in a converted dataset.

This script reads `data/transformer_training_data_converted.npz`, reduces
probability mass on index 0 (PASS) by `reduce_frac` fraction and redistributes
it uniformly across other valid indices, producing `..._augmented.npz`.
"""
import os
import numpy as np

SRC = os.path.join('data', 'transformer_training_data_converted.npz')
DST = os.path.join('data', 'transformer_training_data_augmented.npz')
REDUCE_FRAC = 0.7  # reduce PASS mass to 30% of original


def main():
    if not os.path.exists(SRC):
        print('Source not found:', SRC)
        return
    d = np.load(SRC)
    states = d['states']
    policies = d['policies'].astype(np.float32)
    values = d['values']

    n, dim = policies.shape
    print('Loaded', n, 'samples, dim=', dim)

    # reduce PASS (index 0)
    pass_col = policies[:, 0].copy()
    reduce_amount = pass_col * REDUCE_FRAC
    policies[:, 0] = pass_col - reduce_amount

    # distribute reduced mass uniformly across non-zero indices
    other_indices = list(range(1, dim))
    add_per_index = (reduce_amount / (dim - 1))[:, None]
    policies[:, 1:] += add_per_index

    # renormalize rows
    row_sums = policies.sum(axis=1, keepdims=True)
    zero_mask = (row_sums.squeeze() == 0)
    if zero_mask.any():
        policies[zero_mask, 0] = 1.0
        row_sums = policies.sum(axis=1, keepdims=True)
    policies = policies / row_sums

    # diagnostics
    from collections import Counter
    argmax = np.argmax(policies, axis=1)
    print('Top argmax after augmentation:', Counter(argmax.tolist()).most_common(10))

    np.savez_compressed(DST, states=states, policies=policies, values=values)
    print('Saved augmented dataset to', DST)


if __name__ == '__main__':
    main()
