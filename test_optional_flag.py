#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dm_ai_module as dm
from dm_toolkit import commands_v2 as commands

gs = dm.GameState(seed=42, p0_deck=[1]*40, p1_deck=[91]*40)
gs.execute_command(dm.DrawCardCommand(0, 5))
gs.execute_command(dm.DrawCardCommand(1, 5))
gs.execute_command(dm.TransitionCommand(dm.Phase.MAIN, 0))
gs.execute_command(dm.PlayCardCommand(0, 0, dm.SpawnSource.HAND_SUMMON))

print(f"pending_effects count: {len(gs.pending_effects)}")
if gs.pending_effects:
    pe = gs.pending_effects[0]
    print(f"type: {pe.type} (TRIGGER_ABILITY={dm.EffectType.TRIGGER_ABILITY})")
    print(f"optional: {pe.optional}")
    print(f"controller: {pe.controller}")
    print(f"resolve_type: {pe.resolve_type}")
    
    # コマンド優先で生成（フォールバックを含む）
    import dm_ai_module as _dm
    def _get_legal(gs, card_db):
        try:
            cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
        except Exception:
            cmds = []
        if not cmds:
            try:
                try:
                    cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
                except Exception:
                    try:
                        cmds = commands.generate_legal_commands(gs, card_db) or []
                    except Exception:
                        cmds = []
            except Exception:
                cmds = []
        return cmds

    actions = _get_legal(gs, None)
    print(f"\nTotal actions: {len(actions)}")
    
    pass_count = sum(1 for a in actions if a.type == dm.PlayerIntent.PASS)
    resolve_count = sum(1 for a in actions if a.type == dm.PlayerIntent.RESOLVE_EFFECT)
    
    print(f"PASS actions: {pass_count}")
    print(f"RESOLVE_EFFECT actions: {resolve_count}")
