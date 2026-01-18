#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Introspect dm_ai_module GameState object
"""

import dm_ai_module

gs = dm_ai_module.GameState(42)
gs.setup_test_duel()

print("GameState object attributes:")
attrs = dir(gs)
for attr in attrs:
    if not attr.startswith('_'):
        try:
            val = getattr(gs, attr)
            if not callable(val):
                print(f"  - {attr}: {type(val).__name__} = {val}")
        except:
            print(f"  - {attr}")
