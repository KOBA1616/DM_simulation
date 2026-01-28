#!/usr/bin/env python3
import subprocess, os, csv

VENV_PY = os.path.abspath('.\.venv\Scripts\python.exe')
PTH = os.path.abspath('.')
PYTHON = VENV_PY if os.path.exists(VENV_PY) else 'python'
NATIVE_PYD = os.path.abspath('bin\Release\dm_ai_module.cp312-win_amd64.pyd')
MODEL_A = 'models/best.onnx'
MODEL_B = 'models\\duel_transformer_20260123_030852.onnx'
COMPARE = os.path.abspath('tools\compare_seq_dump.py')
RUN_NATIVE = os.path.abspath('tools\run_head2head_with_native.py')

seeds = list(range(421490, 421500))  # 10 seeds
out_csv = os.path.abspath('logs\seq_parity_batch.csv')

rows = []
for s in seeds:
    native_log = os.path.abspath(f'logs\seq_native_{s}.log')
    pyfb_log = os.path.abspath(f'logs\seq_pyfb_{s}.log')
    # native run
    cmd_native = [PYTHON, RUN_NATIVE, '--pyd', NATIVE_PYD, '--model_a', MODEL_A, '--model_b', MODEL_B, '--games', '1', '--parallel', '1', '--seed', str(s)]
    print('RUN NATIVE:', ' '.join(cmd_native))
    r1 = subprocess.run(cmd_native, cwd=PTH)
    # fallback run (unset native env)
    env = os.environ.copy()
    env.pop('DM_AI_MODULE_NATIVE', None)
    cmd_fallback = [PYTHON, os.path.join(PTH, 'training', 'head2head.py'), MODEL_A, MODEL_B, '--games', '1', '--parallel', '1', '--seed', str(s)]
    print('RUN FALLBACK:', ' '.join(cmd_fallback))
    with open(pyfb_log, 'wb') as fout:
        r2 = subprocess.run(cmd_fallback, cwd=PTH, env=env, stdout=fout, stderr=subprocess.STDOUT)
    # move native stdout to native_log if RUN_NATIVE didn't already redirect
    # run_head2head_with_native uses runpy and doesn't redirect; assume it printed to stdout captured by caller
    # For simplicity, also run native again via CLI but setting DM_AI_MODULE_NATIVE env and redirect
    env_native = os.environ.copy()
    env_native['DM_AI_MODULE_NATIVE'] = NATIVE_PYD
    cmd_native_cli = [PYTHON, os.path.join(PTH, 'training', 'head2head.py'), MODEL_A, MODEL_B, '--games', '1', '--parallel', '1', '--seed', str(s)]
    with open(native_log, 'wb') as fout:
        rn = subprocess.run(cmd_native_cli, cwd=PTH, env=env_native, stdout=fout, stderr=subprocess.STDOUT)
    # compare
    cmp_cmd = [PYTHON, COMPARE, native_log, pyfb_log]
    cp = subprocess.run(cmp_cmd, cwd=PTH, capture_output=True, text=True)
    match = 'OK' in cp.stdout
    rows.append({'seed': s, 'match': match, 'native_log': native_log, 'pyfb_log': pyfb_log, 'cmp_out': cp.stdout.strip()[:200]})

# write csv
with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=['seed','match','native_log','pyfb_log','cmp_out'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('WROTE', out_csv)
