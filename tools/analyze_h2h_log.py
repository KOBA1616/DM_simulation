#!/usr/bin/env python3
import json,sys,csv,collections,argparse

p = argparse.ArgumentParser()
p.add_argument('logfile')
args = p.parse_args()
log = args.logfile

chosen_counts = collections.Counter()
topk_illegal_counts = 0
topk_total = 0
invalid_topk_examples = []
legal_counts_per_turn = []

with open(log, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if 'H2H_JSON:' not in line:
            continue
        try:
            j = line.split('H2H_JSON:',1)[1].strip()
            ev = json.loads(j)
        except Exception:
            continue
        if ev.get('event') == 'chosen_action':
            idx = ev.get('chosen_index')
            chosen_counts[idx] += 1
        if ev.get('event') == 'policy_topk':
            for entry in ev.get('topk',[]):
                topk_total += 1
                if not entry.get('is_legal'):
                    topk_illegal_counts += 1
                    if len(invalid_topk_examples) < 20:
                        invalid_topk_examples.append({'index': entry.get('index'), 'score': entry.get('score')})
        if ev.get('event') == 'legal_map':
            legal_counts_per_turn.append(ev.get('valid_indices', 0))

# write CSV of chosen counts
csv_path = 'logs/h2h_batch100_chosen.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    w = csv.writer(cf)
    w.writerow(['chosen_index','count'])
    for idx,count in chosen_counts.most_common():
        w.writerow([idx,count])

# write summary
summary_path = 'logs/h2h_batch100_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as sf:
    sf.write(f'Total chosen actions: {sum(chosen_counts.values())}\n')
    sf.write(f'Unique chosen indices: {len(chosen_counts)}\n')
    sf.write('Top chosen indices:\n')
    for idx,count in chosen_counts.most_common(10):
        sf.write(f'  {idx}: {count}\n')
    sf.write('\nPolicy top-k illegal fraction: {:.4f} ({}/{})\n'.format(topk_illegal_counts/(topk_total or 1), topk_illegal_counts, topk_total))
    sf.write('\nSample illegal topk examples (up to 20):\n')
    for ex in invalid_topk_examples:
        sf.write(f"  idx={ex['index']} score={ex['score']}\n")
    if legal_counts_per_turn:
        import statistics
        sf.write('\nLegal indices per turn - mean: {:.2f}, median: {}\n'.format(statistics.mean(legal_counts_per_turn), statistics.median(legal_counts_per_turn)))

print('WROTE', csv_path, summary_path)
