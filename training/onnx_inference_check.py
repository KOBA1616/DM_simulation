#!/usr/bin/env python3
"""ONNX inference integrity check
Loads a recent ONNX model, reads a sample state from data, runs inference via onnxruntime,
and prints input/output names, shapes and basic statistics.
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

root = Path(__file__).resolve().parent.parent
models_dir = root / 'models'
data_dir = root / 'data'

# find latest onnx
onnx_files = sorted(models_dir.glob('duel_transformer_*.onnx'), key=lambda p: p.stat().st_mtime, reverse=True)
if not onnx_files:
    print('No ONNX models found in', models_dir)
    sys.exit(2)
onnx_path = onnx_files[0]
print('Using ONNX:', onnx_path)

# load model metadata
m = onnx.load(str(onnx_path))
print('Model opset_import:', [op.version for op in m.opset_import])

# print inputs/outputs
sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
print('Session inputs:')
for i in sess.get_inputs():
    print(' ', i.name, i.shape, i.type)
print('Session outputs:')
for o in sess.get_outputs():
    print(' ', o.name, o.shape, o.type)

# find a sample data npz
npz_files = sorted(data_dir.glob('transformer_training_data_iter*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
if not npz_files:
    print('No data npz files found in', data_dir)
    sys.exit(3)
npz_path = npz_files[0]
print('Using data:', npz_path)

data = np.load(str(npz_path))
states = data['states']
print('states.shape', states.shape, 'dtype', states.dtype)
# pick first sample or batch
sample = states[:1]
# convert to expected dtype (int64)
if sample.dtype != np.int64:
    sample = sample.astype(np.int64)

# Map input names heuristically
inputs = {i.name: sample for i in sess.get_inputs() if 'state' in i.name.lower() or 'input' in i.name.lower()}
# if no state-like input, try first input and broadcast
if not inputs:
    first = sess.get_inputs()[0]
    inputs = {first.name: sample}

# if model expects padding_mask, try to provide one
for i in sess.get_inputs():
    if 'pad' in i.name.lower() or 'mask' in i.name.lower():
        # boolean mask where padding token == 0
        mask = (sample == 0).astype(np.bool_)
        inputs[i.name] = mask

print('Prepared inputs:')
for k,v in inputs.items():
    print(' ', k, np.asarray(v).shape, np.asarray(v).dtype)

# run
outs = sess.run(None, inputs)
print('Outputs count:', len(outs))
for idx, o in enumerate(outs):
    a = np.asarray(o)
    print(f'Output {idx}: shape={a.shape} dtype={a.dtype} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f}')

# basic checks
# if there are two outputs, assume [policy_logits, value]
if len(outs) >= 1:
    p = np.asarray(outs[0])
    print('Policy logits second-dim:', p.shape[1] if p.ndim >= 2 else 'NA')

print('Done')
