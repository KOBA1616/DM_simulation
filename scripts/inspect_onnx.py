import onnxruntime as ort
import sys
p = 'models/duel_transformer_20260123_030635.onnx'
if len(sys.argv)>1:
    p = sys.argv[1]
s = ort.InferenceSession(p)
print('Inputs:')
for i in s.get_inputs():
    print('  name=', i.name, 'shape=', i.shape, 'type=', i.type)
print('Outputs:')
for o in s.get_outputs():
    print('  name=', o.name, 'shape=', o.shape, 'type=', o.type)
