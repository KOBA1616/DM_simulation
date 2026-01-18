import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

card_db = dm.JsonLoader.load_cards('data/cards.json')
gi = dm.GameInstance(0, card_db)
deck = [1]*40
gi.state.set_deck(0, deck[:])
gi.state.set_deck(1, deck[:])
gi.start_game()
print('start active:', gi.state.active_player_id)
print('shields start:', len(gi.state.players[0].shield_zone), len(gi.state.players[1].shield_zone))

ap = gi.state.active_player_id
target = 1-ap
print('attacker', ap, 'target', target)
if len(gi.state.players[target].shield_zone) > 0:
    gi.state.players[target].shield_zone.pop()
else:
    gi.state.winner = dm.GameResult.P1_WIN if ap == 0 else dm.GameResult.P2_WIN

print('after attack shields:', len(gi.state.players[0].shield_zone), len(gi.state.players[1].shield_zone))
print('winner enum:', gi.state.winner)
print('enum values: P1_WIN', int(dm.GameResult.P1_WIN), 'P2_WIN', int(dm.GameResult.P2_WIN), 'NONE', int(dm.GameResult.NONE))
print('winner int:', int(gi.state.winner))
