#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import dm_ai_module as dm

print('Creating collector')
collector = dm.DataCollector()
for episodes in [10, 50, 100, 200]:
    print(f'Collecting {episodes} episodes...')
    batch = collector.collect_data_batch_heuristic(episodes, True, False)
    try:
        n = len(batch.token_states)
    except Exception:
        n = 0
    print(f'  token_states length: {n}')
    try:
        print('  sample shapes (first if available):', getattr(batch, 'token_states', [])[:1])
    except Exception as e:
        print('  error reading sample:', e)
