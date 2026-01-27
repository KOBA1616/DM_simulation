import subprocess
import sys
import argparse


def run_battery(script_path, iterations=200):
    attacks = 0
    for i in range(1, iterations + 1):
        proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        found = False
        if 'ATTACK' in out or 'ATTACK_PLAYER' in out:
            attacks += 1
            found = True
        if i % 50 == 0 or found:
            print(f"run {i}/{iterations} -> attack_found={found}")
    print('Done. iterations=', iterations, 'attacks_detected=', attacks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--iterations', type=int, default=200)
    p.add_argument('--script', default='scripts/test_attack_generation.py')
    args = p.parse_args()
    run_battery(args.script, args.iterations)
