import dm_ai_module as dm

# Setup db and game
 db = dm.CardDatabase()
 db[1] = dm.CardDefinition(1, 'TestCard', 'NONE', [], 0, 1000, dm.CardKeywords(), [])
 game = dm.GameInstance(0, db)
 game.state.set_deck(0, [1]*40)
 game.state.set_deck(1, [1]*40)
 game.start_game()
 for _ in range(10):
     if 'ATTACK' in str(game.state.current_phase).upper():
         break
     dm.PhaseManager.next_phase(game.state, db)

 attacker_iid = 9011
 target_iid = 9012
 game.state.add_test_card_to_battle(0,1,attacker_iid, False, False)
 game.state.add_test_card_to_battle(1,1,target_iid, False, False)

 p = dm.PassiveEffect()
 p.type = dm.PassiveType.ALLOW_ATTACK_UNTAPPED
 p.specific_targets = [attacker_iid]
 p.controller = 0

 try:
     game.state.add_passive_effect(p)
 except Exception as e:
     print('add_passive_effect error', e)

 print('passive count', game.state.get_passive_effect_count())
 for i, eff in enumerate(game.state.passive_effects):
     print(i, eff.type, hasattr(eff, 'specific_targets'), getattr(eff, 'specific_targets', None))
