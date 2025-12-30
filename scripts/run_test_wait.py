import os
import sys
import time
import subprocess

print('PID=' + str(os.getpid()))
print('Sleeping 60s to allow debugger/procdump attach...')
sys.stdout.flush()
# give time to attach interactively
print('Sleeping 300s to allow debugger/procdump attach...')
sys.stdout.flush()
# give time to attach interactively
time.sleep(300)
# run pytest
args = [sys.executable, '-u', '-m', 'pytest', 'tests/ai/test_batch_inference.py::test_batch_inference_basic', '-q', '-s']
print('Running:', ' '.join(args))
sys.stdout.flush()
ret = subprocess.call(args)
print('pytest exit', ret)
