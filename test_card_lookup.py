import dm_ai_module

# Load cards
cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')
print(f"Loaded card database: {len(cdb)} cards")

# Create GameInstance
gi = dm_ai_module.GameInstance(42, cdb)
gs = gi.state

# Setup game
gs.setup_test_duel()
gs.set_deck(0, [1]*40)  # Deck with card ID 1
gs.set_deck(1, [1]*40)

# Start game
dm_ai_module.PhaseManager.start_game(gs, cdb)
dm_ai_module.PhaseManager.fast_forward(gs, cdb)

print(f"\n=== Game State ===")
print(f"Phase: {gs.current_phase}")
print(f"P0 hand size: {len(gs.players[0].hand)}")

# Check hand cards
print(f"\n=== P0 Hand Cards ===")
for i, card in enumerate(gs.players[0].hand):
    print(f"  {i}: instance_id={card.instance_id}, card_id={card.card_id}")
    if card.card_id in cdb:
        card_def = cdb[card.card_id]
        print(f"      -> Found in DB: {card_def.name}, cost={card_def.cost}, type={card_def.type}")
    else:
        print(f"      -> NOT FOUND IN DATABASE!")

# Check card ID 1
print(f"\n=== Card ID 1 in Database ===")
if 1 in cdb:
    card1 = cdb[1]
    print(f"Name: {card1.name}")
    print(f"Cost: {card1.cost}")
    print(f"Type: {card1.type}")
    print(f"Civilizations: {card1.civilizations}")
else:
    print("Card ID 1 NOT FOUND")

# Step to MAIN phase
gi.step()  # MANA_CHARGE
gi.step()  # PASS

print(f"\n=== After stepping to MAIN ===")
print(f"Phase: {gs.current_phase}")
print(f"P0 mana zone: {len(gs.players[0].mana_zone)}")

# Check actions in MAIN
from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(gs, cdb, strict=False)
print(f"Commands in MAIN: {len(actions) if actions is not None else 0}")
