import sys, os
sys.path.insert(0, os.getcwd())
import dm_ai_module
try:
	from dm_toolkit import commands_v2
	generate_legal_commands = commands_v2.generate_legal_commands
except Exception:
	generate_legal_commands = None

# Setup card_db as in test
card_db = {}
card_db[100] = dm_ai_module.CardDefinition()
card_db[100].keywords.meta_counter_play = True
card_db[100].type = dm_ai_module.CardType.CREATURE
card_db[100].cost = 8

card_db[99] = dm_ai_module.CardDefinition()
card_db[99].cost = 0
card_db[99].type = dm_ai_module.CardType.CREATURE

# Setup game
game = dm_ai_module.GameState(1000)
game.active_player_id = 0
game.current_phase = dm_ai_module.Phase.MAIN
p0 = game.players[0]
p1 = game.players[1]
# add zero cost to p0 hand
game.add_card_to_hand(0, 99, 1)
# add meta to p1 hand
game.add_card_to_hand(1, 100, 2)
# add mana and play 0 cost card
game.add_card_to_mana(0, 99, 3)
# Generate action and play
cmds = generate_legal_commands(game, card_db) if generate_legal_commands else []
try:
	print('commands:', [c.to_dict() for c in cmds])
except Exception:
	print('commands: (unable to to_dict) count=', len(cmds))

# Best-effort: find DECLARE_PLAY by card id among commands
play_cmd = None
for c in cmds:
	try:
		d = c.to_dict()
		t = getattr(d.get('type'), 'name', d.get('type'))
		if str(t).upper().endswith('DECLARE_PLAY') and d.get('card_id') == 99:
			play_cmd = c
			break
	except Exception:
		continue

if play_cmd is not None:
	try:
		game.execute_command(play_cmd)
	except Exception:
		try:
			play_cmd.execute(game)
		except Exception:
			# Last resort: try compatibility wrapper against legacy action if available
			try:
				from dm_toolkit.compat_wrappers import execute_action_compat
				execute_action_compat(game, None, card_db)
			except Exception:
				pass
# pay cost
actions = generate_legal_commands(game, card_db) if generate_legal_commands else []
print('after declare, actions:', [(getattr(a,'type',None), getattr(a,'card_id',None)) for a in actions])
pay = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
print('pay', pay)
pay_cmd = getattr(pay, 'command', None) if pay is not None else None
if pay_cmd is not None:
	try:
		game.execute_command(pay_cmd)
	except Exception:
		try:
			pay_cmd.execute(game)
		except Exception:
				try:
					from dm_toolkit.compat_wrappers import execute_action_compat
					execute_action_compat(game, pay, card_db)
				except Exception:
						try:
							from dm_toolkit.compat_wrappers import execute_action_compat
							execute_action_compat(game, pay, card_db)
						except Exception:
							dm_ai_module.GameLogicSystem.resolve_action(game, pay, card_db)
else:
	try:
		from dm_toolkit.compat_wrappers import execute_action_compat
		execute_action_compat(game, pay, card_db)
	except Exception:
		dm_ai_module.GameLogicSystem.resolve_action(game, pay, card_db)
# resolve
actions = generate_legal_commands(game, card_db) if generate_legal_commands else []
print('before resolve, actions:', [(getattr(a,'type',None), getattr(a,'card_id',None)) for a in actions])
res = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
print('resolve', res)
res_cmd = getattr(res, 'command', None) if res is not None else None
if res_cmd is not None:
	try:
		game.execute_command(res_cmd)
	except Exception:
		try:
			res_cmd.execute(game)
		except Exception:
				try:
					from dm_toolkit.compat_wrappers import execute_action_compat
					execute_action_compat(game, res, card_db)
				except Exception:
						try:
							from dm_toolkit.compat_wrappers import execute_action_compat
							execute_action_compat(game, res, card_db)
						except Exception:
							dm_ai_module.GameLogicSystem.resolve_action(game, res, card_db)
else:
	try:
		from dm_toolkit.compat_wrappers import execute_action_compat
		execute_action_compat(game, res, card_db)
	except Exception:
		dm_ai_module.GameLogicSystem.resolve_action(game, res, card_db)
print('turn_stats played_without_mana:', getattr(game.turn_stats, 'played_without_mana', None))
# advance phases
game.current_phase = dm_ai_module.Phase.ATTACK
print('before next_phase: current_phase=', game.current_phase)
print('played_without_mana=', getattr(game.turn_stats, 'played_without_mana', None))
print('opponent hand ids=', [getattr(c,'card_id',None) for c in game.players[1].hand])
print('card_db keys=', list(card_db.keys()))
import inspect
try:
	print('PhaseManager module:', dm_ai_module.PhaseManager.__module__)
	print('PhaseManager file:', inspect.getsourcefile(dm_ai_module.PhaseManager))
except Exception:
	pass

dm_ai_module.PhaseManager.next_phase(game, card_db)
print('after next_phase: current_phase=', game.current_phase)
print('pending_effects:', game.get_pending_effects_info())
