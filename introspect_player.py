#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introspect dm_ai_module Player object attributes
"""

import dm_ai_module

gs = dm_ai_module.GameState(42)
gs.setup_test_duel()

player = gs.players[0]

print("Player object attributes:")
attrs = dir(player)
for attr in attrs:
    if not attr.startswith('_'):
        print(f"  - {attr}")
        try:
            val = getattr(player, attr)
            if not callable(val):
                print(f"      value: {val}")
        except:
            pass
