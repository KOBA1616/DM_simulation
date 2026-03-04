"""SELECT_NUMBERコマンドの属性調査スクリプト"""
import sys
import os
sys.path.insert(0, 'bin/Release')
os.environ['DM_DISABLE_NATIVE'] = '0'

import dm_ai_module

db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
gi = dm_ai_module.GameInstance(7, db)
s = gi.state
s.set_deck(0, [1]*40)
s.set_deck(1, [1]*40)
gi.start_game()

print(f"=== 初期状態 ===", flush=True)
print(f"phase: {getattr(s,'current_phase','?')}", flush=True)
print(f"turn: {getattr(s,'turn_count','?')}", flush=True)

found_select = False
for step in range(200):
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
    ct_list = [str(c.type).upper() for c in legal]

    if any('SELECT' in t for t in ct_list):
        sel = [c for c in legal if 'SELECT' in str(c.type).upper()][0]
        print(f"\n=== FOUND SELECT at step {step} ===", flush=True)
        print(f"  type: {sel.type}", flush=True)
        all_attrs = [a for a in dir(sel) if not a.startswith('_')]
        print(f"  attrs: {all_attrs}", flush=True)
        for a in all_attrs:
            try:
                print(f"   .{a} = {getattr(sel, a)}", flush=True)
            except Exception as e:
                print(f"   .{a} => ERROR: {e}", flush=True)

        # pending_query調査
        pq = getattr(s, 'pending_query', None)
        print(f"\n  pending_query: {pq}", flush=True)
        if pq is not None:
            pq_attrs = [a for a in dir(pq) if not a.startswith('_')]
            print(f"  pq attrs: {pq_attrs}", flush=True)
            for a in pq_attrs:
                try:
                    print(f"   pq.{a} = {getattr(pq, a)}", flush=True)
                except Exception as e:
                    print(f"   pq.{a} => ERROR: {e}", flush=True)

        # value=0で応答を試みる
        print(f"\n--- value=0で応答テスト ---", flush=True)
        try:
            sel.value = 0
            print(f"  sel.value = 0 設定 OK", flush=True)
        except Exception as e:
            print(f"  sel.value = 0 設定 NG: {e}", flush=True)

        # 全legal commandsの型を表示
        print(f"\n  全legal types: {ct_list}", flush=True)
        found_select = True
        break

    elif any('PLAY' in t for t in ct_list):
        c = [c for c in legal if 'PLAY' in str(c.type).upper()][0]
        gi.resolve_command(c)
        bz = len(s.players[0].battle_zone) if hasattr(s, 'players') else '?'
        mz = len(s.players[0].mana_zone) if hasattr(s, 'players') else '?'
        print(f"step {step}: PLAY executed  [bz={bz} mana={mz}]", flush=True)
    elif any('MANA' in t for t in ct_list):
        c = [c for c in legal if 'MANA' in str(c.type).upper()][0]
        gi.resolve_command(c)
        print(f"step {step}: MANA charged", flush=True)
    elif any('PASS' in t for t in ct_list):
        c = [c for c in legal if 'PASS' in str(c.type).upper()][0]
        gi.resolve_command(c)
        print(f"step {step}: PASS  phase={getattr(s,'current_phase','?')}", flush=True)
    else:
        dm_ai_module.PhaseManager.next_phase(s, db)
        print(f"step {step}: next_phase  phase={getattr(s,'current_phase','?')}", flush=True)

if not found_select:
    print("SELECT not found in 200 steps", flush=True)
