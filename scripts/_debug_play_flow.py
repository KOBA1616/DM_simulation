from dm_ai_module import GameState, DeclarePlayCommand, PayCostCommand, ResolvePlayCommand, CardDefinition, CardType

state = GameState()
state._ensure_player(0)
state._ensure_player(1)
state.add_card_to_hand(0, card_id=1, instance_id=1)
# register card def
_CARD_REGISTRY = getattr(__import__('dm_ai_module'), '_CARD_REGISTRY', None)
import dm_ai_module as dm
if not hasattr(dm, '_CARD_REGISTRY'):
    dm._CARD_REGISTRY = {}
dm._CARD_REGISTRY[1] = CardDefinition(1, name='TestCreature', civilization=None, civilizations=[], races=[], cost=1, power=1, keywords=None, effects=[])
dm._CARD_REGISTRY[1].type = CardType.CREATURE
# add mana
state.add_card_to_mana(0, card_id=200, instance_id=100)
print('initial hand:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None)) for c in state.players[0].hand])
print('initial mana:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None)) for c in state.players[0].mana_zone])
# declare
declare = DeclarePlayCommand(player_id=0, card_id=1, source_instance_id=1)
declare.execute(state)
print('after declare hand:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None)) for c in state.players[0].hand])
print('stack_zone:', getattr(state,'stack_zone',None))
# pay
pay = PayCostCommand(player_id=0, amount=1)
pay.execute(state)
print('after pay mana_zone:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None), getattr(c,'is_tapped',None)) for c in state.players[0].mana_zone])
print('stack top paid:', getattr(state,'stack_zone',None) and getattr(state.stack_zone[-1],'paid',None))
# resolve
resolve = ResolvePlayCommand(player_id=0, card_id=1, card_def=None)
resolve.execute(state)
print('after resolve battle_zone:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None)) for c in state.players[0].battle_zone])
print('graveyard:', [(getattr(c,'card_id',None), getattr(c,'instance_id',None)) for c in state.players[0].graveyard])
print('stack_zone now:', getattr(state,'stack_zone',None))
