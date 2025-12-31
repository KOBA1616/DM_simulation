import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'bin'))
import dm_ai_module
cmd_def = dm_ai_module.CommandDef()
print('dir(CommandDef):')
print([n for n in dir(cmd_def) if not n.startswith('_')])
# try setting some fields
for name in ['instance_id','from_zone','to_zone','amount','target_instance','owner_id','mutation_kind']:
    print(name, 'exists?', hasattr(cmd_def, name))
    try:
        val = getattr(cmd_def, name)
        print('  current:', val)
    except Exception as e:
        print('  getattr error', e)

print('CommandDef repr:', cmd_def)
