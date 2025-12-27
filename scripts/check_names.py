import sys, os
sys.path.insert(0, os.getcwd())
import dm_ai_module
names=['GameState','ActionGenerator','EffectResolver','PhaseManager','Action','ActionType','EffectType','Phase','SpawnSource','CardData','GameResult','CardDefinition','Civilization','CardType']
for n in names:
    print(n, hasattr(dm_ai_module, n))
