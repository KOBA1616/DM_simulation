import json
from collections import Counter, defaultdict
import statistics

path = "logs/runner_debug.jsonl"
counts = Counter()
turns = defaultdict(list)
hand_p1 = defaultdict(list)
hand_p2 = defaultdict(list)
seeds = defaultdict(list)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        w = obj.get("winner", 0)
        counts[w] += 1
        turns[w].append(obj.get("turn", 0))
        hand_p1[w].append(obj.get("p1_hand", 0))
        hand_p2[w].append(obj.get("p2_hand", 0))
        seeds[w].append(obj.get("seed"))

print("Counts:")
for k in sorted(counts.keys()):
    print(f"  winner={k}: {counts[k]}")

print("\nTurn stats:")
for k in sorted(turns.keys()):
    t = turns[k]
    print(f"  winner={k}: mean={statistics.mean(t):.2f} median={statistics.median(t):.2f} max={max(t)} min={min(t)}")

print("\nHand size stats (p1/p2):")
for k in sorted(hand_p1.keys()):
    p1 = hand_p1[k]
    p2 = hand_p2[k]
    print(f"  winner={k}: p1_mean={statistics.mean(p1) if p1 else 0:.2f} p2_mean={statistics.mean(p2) if p2 else 0:.2f}")

print("\nAdditional checks:")
# fraction of winner==1 with turn == max_turn
if 1 in turns:
    t = turns[1]
    max_turn = max(t)
    frac = sum(1 for x in t if x==max_turn)/len(t)
    print(f"  winner=1: max_turn={max_turn}, fraction_at_max={frac:.2f} ({sum(1 for x in t if x==max_turn)}/{len(t)})")

# sample seeds for winner=1
if 1 in seeds:
    print(f"  sample seeds winner=1 (first 10): {seeds[1][:10]}")

# check how many games have p1_hand or p2_hand > 20 (indicates odd large hand)
big_hand_counts = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("p1_hand",0) > 20 or obj.get("p2_hand",0) > 20:
            big_hand_counts += 1
print(f"\nGames with unusual large hand (>20): {big_hand_counts}")
