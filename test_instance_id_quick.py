# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'c:\Users\ichirou\DM_simulation\bin\Release')

import dm_ai_module

print("Quick test of instance_id fix...")
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()

# P0のデッキを設定
deck0 = [1] * 40
gs.set_deck(0, deck0)
print(f"P0 deck[0].instance_id = {gs.players[0].deck[0].instance_id}")
print(f"P0 deck[39].instance_id = {gs.players[0].deck[39].instance_id}")

# P1のデッキを設定
deck1 = [1] * 40
gs.set_deck(1, deck1)
print(f"P1 deck[0].instance_id = {gs.players[1].deck[0].instance_id}")
print(f"P1 deck[39].instance_id = {gs.players[1].deck[39].instance_id}")

if gs.players[1].deck[0].instance_id == 40:
    print("✓ PASS: P1 starts at instance_id 40!")
else:
    print(f"✗ FAIL: P1 starts at instance_id {gs.players[1].deck[0].instance_id}, expected 40")
