import dm_ai_module as m

state = m.GameState(100)
native = getattr(state, '_native', state)
print(f"wrapper id={id(state)} type={type(state)} native id={id(native)} type={type(native)}")

# initialize
try:
    db = m.CardRegistry.get_all_cards()
except Exception:
    db = {}
m.initialize_card_stats(native, db, 100)

native.add_card_to_deck(0, 1, 10)
native.add_card_to_deck(0, 1, 11)

card_data = m.CardData(
    1, "Test Card", 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], []
)
m.register_card_data(card_data)

native.add_card_to_mana(0, 1, 0)

action = m.ActionDef()
action.type = m.EffectPrimitive.IF
action.target_player = "PLAYER_SELF"

filter_def = m.FilterDef()
filter_def.zones = ["MANA_ZONE"]
filter_def.civilizations = m.CivilizationList([m.Civilization.FIRE])
action.filter = filter_def

then_act = m.ActionDef()
then_act.type = m.EffectPrimitive.DRAW_CARD
then_act.value1 = 1

action.options = [[then_act]]

print("BEFORE: wrapper_players_id=", id(state.players), "native_players_id=", id(native.players))
try:
    wp = state.players[0]
    print("wrapper player type", type(wp), "id", id(wp))
    try:
        print("wrapper._p id", id(getattr(wp, '_p', None)))
    except Exception:
        pass
except Exception as e:
    print("failed inspect wrapper players", e)

print("BEFORE lengths: wrapper hand", len(state.players[0].hand), "native hand", len(native.players[0].hand))
print('GenericCardSystem.resolve_action=', m.GenericCardSystem.resolve_action)
print('native._wrapper=', getattr(native, '_wrapper', None), 'wrapper id=', id(getattr(native, '_wrapper', None)))

m.GenericCardSystem.resolve_action(native, action, 0)

print("AFTER lengths: wrapper hand", len(state.players[0].hand), "native hand", len(native.players[0].hand))

try:
    # Direct draw to verify draw_cards functionality
    native.draw_cards(0, 1)
except Exception as e:
    print('direct draw failed', e)
print("AFTER DIRECT draw lengths: wrapper hand", len(state.players[0].hand), "native hand", len(native.players[0].hand))
print('Done')

print('\n-- Inspecting implicit filter evaluation manually --')
cond = filter_def
print('cond.zones=', getattr(cond,'zones',None))
print('cond.civilizations=', getattr(cond,'civilizations',None))
try:
    civs = getattr(cond, 'civilizations', []) or []
    civ_vals = set()
    for c in civs:
        try:
            if hasattr(c, 'value'):
                civ_vals.add(int(getattr(c, 'value')))
            else:
                civ_vals.add(int(c))
        except Exception:
            pass
    print('civ_vals=', civ_vals)
except Exception as e:
    print('civ parse error', e)

zone_map = {
    'MANA_ZONE': 'mana_zone',
    'HAND': 'hand',
    'BATTLE_ZONE': 'battle_zone',
    'DECK': 'deck',
    'SHIELD_ZONE': 'shield_zone',
    'GRAVEYARD': 'graveyard',
}
for z in getattr(cond, 'zones', []):
    zname = str(z).upper()
    target_attr = zone_map.get(zname, None)
    print('zone', z, '->', target_attr)
    native_zone = getattr(native.players[0], target_attr, None)
    print('native_zone repr:', repr(native_zone))
    for idx, c in enumerate(list(native_zone) if native_zone is not None else []):
        try:
            cid = None
            if isinstance(c, int):
                cid = int(c)
            else:
                cid = getattr(c, 'card_id', getattr(c, 'id', None))
        except Exception:
            cid = None
        print(' element', idx, 'type', type(c), 'cid', cid)

print('Done')
try:
    mp = getattr(m, '_NATIVE_TO_WRAPPER', None)
    print('\n_module _NATIVE_TO_WRAPPER keys sample:')
    if mp is None:
        print(' NO MAP')
    else:
        try:
            for k in list(mp.keys())[:10]:
                try:
                    print(' key=', repr(k), ' -> ', repr(mp[k]))
                except Exception:
                    print(' key_repr_failed')
        except Exception:
            print('map read failed')
except Exception as e:
    print('map inspect failed', e)

try:
    # Inspect wrapper via mapping when available
    wrapped = None
    try:
        if mp is not None:
            wrapped = mp.get(id(native)) or mp.get(native) or mp.get(id(state)) or mp.get(state)
    except Exception:
        wrapped = None
    print('\nmapped wrapper found:', wrapped)
    if wrapped is not None:
        try:
            print('wrapped.players id=', id(getattr(wrapped, 'players', None)), 'type=', type(getattr(wrapped, 'players', None)))
            for i, wp in enumerate(getattr(wrapped, 'players', [])):
                try:
                    print('wp', i, 'id=', id(wp), '_p=', getattr(wp, '_p', None), 'hand_len=', len(getattr(wp, 'hand', [])))
                except Exception as e:
                    print('wp inspect failed', e)
        except Exception as e:
            print('wrapped inspect error', e)
except Exception as e:
    print('mapped wrapper check failed', e)
