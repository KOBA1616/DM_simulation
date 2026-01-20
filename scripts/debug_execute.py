import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module
s = dm_ai_module.GameState(100)
s.setup_test_duel()
def log(msg):
	with open(os.path.join(os.path.dirname(__file__), 'debug.log'), 'a', encoding='utf-8') as f:
		f.write(msg + '\n')

log('players: ' + str(len(s.players)))
cs = s.add_test_card_to_battle(0, 999, 0, False, False)
log('added card obj: ' + repr(cs) + ' instance_id:' + str(getattr(cs,'instance_id',None)))
inst = s.get_card_instance(0)
log('get_card_instance returned: ' + repr(inst) + ' is_tapped:' + str(getattr(inst,'is_tapped',None)))
cmd = dm_ai_module.MutateCommand(0, dm_ai_module.MutationType.TAP)
log('cmd: ' + repr(getattr(cmd,'type',None)) + ' ' + str(getattr(cmd,'instance_id',None)) + ' ' + str(getattr(cmd,'value',None)))
s.execute_command(cmd)
inst2 = s.get_card_instance(0)
log('after execute, instance: ' + repr(inst2) + ' is_tapped:' + str(getattr(inst2,'is_tapped',None)))
log('same object? ' + str(inst is inst2))
