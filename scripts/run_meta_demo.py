import sys, os
sys.path.insert(0, os.getcwd())
import dm_ai_module
try:
	from dm_toolkit.commands_new import generate_legal_commands
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
actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
print('actions:', [(a.type, a.card_id) for a in actions])
if generate_legal_commands:
	cmds = generate_legal_commands(game, card_db)
	try:
		print('commands:', [c.to_dict() for c in cmds])
	except Exception:
		print('commands: (unable to to_dict) count=', len(cmds))
else:
	cmds = []

play_action = next((a for a in actions if a.type == dm_ai_module.ActionType.DECLARE_PLAY and a.card_id == 99), None)
print('play_action', play_action)
# Prefer command execution when available
play_cmd = getattr(play_action, 'command', None) if play_action is not None else None
if play_cmd is None and cmds:
	# best-effort find matching command by card id in dict repr
	for c in cmds:
		try:
			d = c.to_dict()
			if any(str(99) in str(v) for v in d.values()):
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
			dm_ai_module.EffectResolver.resolve_action(game, play_action, card_db)
else:
	dm_ai_module.EffectResolver.resolve_action(game, play_action, card_db)
# pay cost
actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
print('after declare, actions:', [(a.type, a.card_id) for a in actions])
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
			dm_ai_module.EffectResolver.resolve_action(game, pay, card_db)
else:
	dm_ai_module.EffectResolver.resolve_action(game, pay, card_db)
# resolve
actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
print('before resolve, actions:', [(a.type, a.card_id) for a in actions])
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
			dm_ai_module.EffectResolver.resolve_action(game, res, card_db)
else:
	dm_ai_module.EffectResolver.resolve_action(game, res, card_db)
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
