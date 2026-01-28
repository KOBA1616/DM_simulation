#!/usr/bin/env python3
import os, sys, runpy, argparse

p = argparse.ArgumentParser()
p.add_argument('--pyd', required=True)
p.add_argument('--model_a', required=True)
p.add_argument('--model_b', required=True)
p.add_argument('--games', type=int, default=1)
p.add_argument('--parallel', type=int, default=1)
p.add_argument('--seed', type=int, default=None)
args = p.parse_args()

# Ensure native pyd is visible
os.environ['DM_AI_MODULE_NATIVE'] = args.pyd
# Prepend the directory to sys.path so import finds the pyd first
pyd_dir = os.path.dirname(args.pyd)
if pyd_dir not in sys.path:
    sys.path.insert(0, pyd_dir)

# Build argv for head2head
argv = ["training/head2head.py", args.model_a, args.model_b, '--games', str(args.games), '--parallel', str(args.parallel)]
if args.seed is not None:
    argv += ['--seed', str(args.seed)]

# Set sys.argv and run
sys.argv = argv
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'training', 'head2head.py'), run_name='__main__')
