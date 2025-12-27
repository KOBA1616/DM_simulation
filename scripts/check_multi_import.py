import sys, os, traceback
sys.path.append(os.path.join(os.path.dirname('tests/integration/test_meta_counter.py'),'../../bin'))
try:
    import dm_ai_module
    print('dm module ok')
    try:
        from dm_ai_module import GameState, ActionGenerator, EffectResolver, PhaseManager, Action, ActionType, EffectType, Phase, SpawnSource, CardData, GameResult, CardDefinition, Civilization, CardType
        print('all names imported OK')
    except Exception as e:
        print('IMPORT ERROR:', type(e), e)
        traceback.print_exc()
except Exception:
    traceback.print_exc()
