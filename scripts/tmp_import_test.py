import os
print('before prepend PATH[0]=', os.environ['PATH'].split(';')[0])
# Prepend ONNX lib path
p = r'C:\Users\ichirou\DM_simulation\build_mingw_onnx\onnxruntime-1.18.0\onnxruntime-win-x64-1.18.0\lib'
os.environ['PATH'] = p + ';' + os.environ['PATH']
print('after prepend PATH[0]=', os.environ['PATH'].split(';')[0])
try:
    import dm_ai_module as m
    print('import succeeded, has NeuralEvaluator=', hasattr(m, 'NeuralEvaluator'))
except Exception as e:
    print('import failed:', e)
    raise
