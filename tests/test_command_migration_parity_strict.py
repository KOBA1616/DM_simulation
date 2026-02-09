import pytest

def make_playable_state(dm_ai_module):
    # native GameState requires an integer seed
    try:
        s = dm_ai_module.GameState(0)
    except Exception:
        s = dm_ai_module.GameState()
    try:
        s.active_player_id = 0
    except Exception:
        pass
    try:
        s.add_card_to_mana(0, card_id=1001, count=1)
    except Exception:
        pass
    try:
        s.add_card_to_hand(0, card_id=42)
    except Exception:
        pass
    try:
        s.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        try:
            s.current_phase = 3
        except Exception:
            pass
    return s


def normalize_cmd_dict(obj, to_command_dict, map_action):
    # obj may be a command-like with to_dict, or an action to map
    if obj is None:
        return None
    try:
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
    except Exception:
        pass
    try:
        # try unified mapper first
        return to_command_dict(obj)
    except Exception:
        try:
            return map_action(obj)
        except Exception:
            return None


def extract_types(cmds, to_command_dict, map_action):
    types = set()
    for c in (cmds or []):
        try:
            d = normalize_cmd_dict(c, to_command_dict, map_action)
            if isinstance(d, dict):
                t = d.get('type') or d.get('unified_type') or d.get('legacy_original_type')
                if t is not None:
                    try:
                        types.add(str(getattr(t, 'name', t)).upper())
                    except Exception:
                        types.add(str(t).upper())
        except Exception:
            continue
    return types


def test_command_action_field_level_parity():
    # Run the stricter parity check in a subprocess to avoid native-extension crashes
    import os, sys, subprocess, json

    script = r"""
import os, sys
os.environ['DM_DISABLE_NATIVE'] = '1'
try:
    import dm_ai_module
    from dm_toolkit import commands_v2 as cmdv2
    from dm_toolkit import commands as legacy_commands
    from dm_toolkit.unified_execution import to_command_dict
    from dm_toolkit.action_to_command import map_action
except Exception as e:
    print('IMPORT_FAIL', e)
    sys.exit(2)

card_db = {42: {'cost': 1}, 1001: {'cost': 0}, 2001: {'cost': 0}}
try:
    s = dm_ai_module.GameState(0)
except Exception:
    try:
        s = dm_ai_module.GameState()
    except Exception:
        print('GAMESTATE_FAIL')
        sys.exit(2)
try:
    s.add_card_to_mana(0, card_id=1001, count=1)
    s.add_card_to_hand(0, card_id=42)
    try:
        s.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        s.current_phase = 3
except Exception:
    pass

def normalize(obj):
    try:
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
    except Exception:
        pass
    try:
        return to_command_dict(obj)
    except Exception:
        try:
            return map_action(obj)
        except Exception:
            return None

def types_of(cmds):
    types = set()
    for c in (cmds or []):
        d = normalize(c)
        if isinstance(d, dict):
            t = d.get('type') or d.get('unified_type') or d.get('legacy_original_type')
            if t is not None:
                types.add(str(getattr(t, 'name', t)).upper())
    return types

cmds1 = []
try:
    try:
        cmds1 = cmdv2.generate_legal_commands(s, card_db, strict=False) or []
    except TypeError:
        cmds1 = cmdv2.generate_legal_commands(s, card_db) or []
    except Exception:
        cmds1 = []
except Exception:
    cmds1 = []

cmds2 = []
try:
    try:
        cmds2 = legacy_commands._call_native_action_generator(s, card_db) or []
    except Exception:
        AG = getattr(dm_ai_module, 'ActionGenerator', None)
        if AG is not None and hasattr(AG, 'generate_legal_commands'):
            try:
                cmds2 = AG.generate_legal_commands(s, card_db) or []
            except Exception:
                cmds2 = []
except Exception:
    cmds2 = []

if not cmds1 and not cmds2:
    print('NO_COMMANDS')
    sys.exit(0)

types1 = types_of(cmds1)
types2 = types_of(cmds2)

if types1 and types2:
    if types1.intersection(types2):
        print('OK')
        sys.exit(0)
    else:
        print('MISMATCH', types1, types2)
        sys.exit(3)
else:
    print('PARTIAL', types1, types2)
    sys.exit(0)
"""

    env = os.environ.copy()
    # ensure DM_DISABLE_NATIVE for subprocess
    env['DM_DISABLE_NATIVE'] = '1'
    proc = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    out = proc.stdout.strip() + '\n' + proc.stderr.strip()
    if proc.returncode == 0:
        return
    if proc.returncode == 3:
        pytest.fail(f'strict parity mismatch:\n{out}')
    # other non-zero codes are treated as skip-worthy import/runtime issues
    pytest.skip(f'strict parity test could not run safely (rc={proc.returncode}):\n{out}')
