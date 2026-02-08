"""Check card placement during GUI session initialization."""

from dm_toolkit.gui.game_session import GameSession
import dm_ai_module

# Create session
session = GameSession()

# Load card database
try:
    card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
except Exception as e:
    print(f"Error loading card database: {e}")
    card_db = None

# Initialize game
print("Initializing game...")
session.initialize_game(card_db)

# Check state
if session.gs:
    print("\n✅ Game initialized successfully")
    print(f"\nGame State:")
    print(f"  Turn: {session.gs.turn_number}")
    print(f"  Active player: {session.gs.active_player_id}")
    print(f"  Phase: {session.gs.current_phase}")
    
    print(f"\nPlayer 0:")
    p0 = session.gs.players[0]
    print(f"  Hand: {len(p0.hand)} cards")
    print(f"  Mana zone: {len(p0.mana_zone)} cards")
    print(f"  Battle zone: {len(p0.battle_zone)} cards")
    print(f"  Shield zone: {len(p0.shield_zone)} cards")
    print(f"  Graveyard: {len(p0.graveyard)} cards")
    print(f"  Deck: {len(p0.deck)} cards")
    
    print(f"\nPlayer 1:")
    p1 = session.gs.players[1]
    print(f"  Hand: {len(p1.hand)} cards")
    print(f"  Mana zone: {len(p1.mana_zone)} cards")
    print(f"  Battle zone: {len(p1.battle_zone)} cards")
    print(f"  Shield zone: {len(p1.shield_zone)} cards")
    print(f"  Graveyard: {len(p1.graveyard)} cards")
    print(f"  Deck: {len(p1.deck)} cards")
    
    # Check total
    p0_total = (len(p0.hand) + len(p0.mana_zone) + len(p0.battle_zone) +
                len(p0.shield_zone) + len(p0.graveyard) + len(p0.deck))
    p1_total = (len(p1.hand) + len(p1.mana_zone) + len(p1.battle_zone) +
                len(p1.shield_zone) + len(p1.graveyard) + len(p1.deck))
    
    print(f"\nTotal cards:")
    print(f"  Player 0: {p0_total} cards")
    print(f"  Player 1: {p1_total} cards")
else:
    print("❌ Game initialization failed")
