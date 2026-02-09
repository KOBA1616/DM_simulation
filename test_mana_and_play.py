"""Check mana availability and playability in MAIN phase."""
import sys
sys.path.insert(0, '.')
import dm_ai_module

seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)
gs = game_instance.state

gs.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f"=== Turn {gs.turn_number}, Phase: {gs.current_phase} ===\n")

# Execute 1 MANA_CHARGE for P0
from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(gs, card_db, strict=False)
mana_action = next((a for a in actions if int(a.type) == 1), None)  # MANA_CHARGE

if mana_action:
    print(f"Executing MANA_CHARGE for card_id={mana_action.card_id}")
    game_instance.resolve_action(mana_action)
    print(f"After MANA_CHARGE: phase={gs.current_phase}, mana_zone={len(gs.players[0].mana_zone)}")
    
    # Check mana zone cards
    print(f"\nP0 Mana Zone:")
    for i, card in enumerate(gs.players[0].mana_zone):
        print(f"  [{i}] instance_id={card.instance_id}, card_id={card.card_id}, is_tapped={card.is_tapped}")
    
    # Execute PASS to progress to MAIN
    dm_ai_module.PhaseManager.fast_forward(gs, card_db)
    print(f"\nAfter fast_forward: phase={gs.current_phase}, turn={gs.turn_number}")
    
    # Check actions in MAIN phase
    actions_main = commands.generate_legal_commands(gs, card_db, strict=False)
    print(f"\nMAIN_PHASE actions: {len(actions_main)}")
    
    if gs.current_phase == dm_ai_module.Phase.MAIN:
        # Check available mana
        p0 = gs.players[0]
        print(f"\nP0 Status:")
        print(f"  Hand: {len(p0.hand)} cards")
        print(f"  Mana Zone: {len(p0.mana_zone)} cards")
        print(f"  Untapped mana: {sum(1 for c in p0.mana_zone if not c.is_tapped)}")
        
        # List hand cards
        print(f"\nP0 Hand:")
        for i, card in enumerate(p0.hand):
            card_def = card_db.get(card.card_id)
            if card_def:
                cost = card_def.get('cost', 0)
                print(f"  [{i}] card_id={card.card_id}, name={card_def.get('name', 'Unknown')}, cost={cost}")
        
        # Check if we can play any card
        untapped_mana = sum(1 for c in p0.mana_zone if not c.is_tapped)
        playable = [c for c in p0.hand if card_db.get(c.card_id, {}).get('cost', 99) <= untapped_mana]
        print(f"\nPlayable cards with {untapped_mana} mana: {len(playable)}")
        
        if len(actions_main) == 0 and len(playable) > 0:
            print("  => BUG: Playable cards exist but no PLAY actions generated!")
        elif len(actions_main) == 0:
            print("  => EXPECTED: No mana available to play cards")
    else:
        print(f"Not in MAIN phase: {gs.current_phase}")
else:
    print("No MANA_CHARGE action available")
