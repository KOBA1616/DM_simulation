"""Debug MAIN phase action generation step by step."""
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
print(f"After start_game: phase={gs.current_phase}, turn={gs.turn_number}")

# Fast-forward to MANA phase
dm_ai_module.PhaseManager.fast_forward(gs, card_db)
print(f"After fast_forward: phase={gs.current_phase}, turn={gs.turn_number}")

# Execute 1 MANA_CHARGE + PASS sequence manually
print(f"\n=== Executing MANA_CHARGE ===")
actions1 = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
print(f"Actions in MANA phase: {len(actions1)}")
mana_action = next((a for a in actions1 if int(a.type) == 1), None)

if mana_action:
    game_instance.resolve_action(mana_action)
    print(f"Executed MANA_CHARGE for card_id={mana_action.card_id}")
    print(f"After MANA_CHARGE: phase={gs.current_phase}")
    
    # Now get actions again - should include PASS
    print(f"\n=== After MANA_CHARGE, checking actions ===")
    actions2 = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
    print(f"Actions available: {len(actions2)}")
    for i, a in enumerate(actions2[:10]):
        print(f"  [{i}] type={a.type} (int={int(a.type)}), card_id={a.card_id}")
    
    # Find and execute PASS (type=0)
    pass_action = next((a for a in actions2 if int(a.type) == 0), None)
    if pass_action:
        print(f"\n=== Executing PASS ===")
        game_instance.resolve_action(pass_action)
        print(f"After PASS: phase={gs.current_phase}, turn={gs.turn_number}")
        
        # Fast-forward to see what happens
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        print(f"After fast_forward: phase={gs.current_phase}, turn={gs.turn_number}")
        
        # Check actions in current phase
        actions3 = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
        print(f"\nActions in {gs.current_phase}: {len(actions3)}")
        for i, a in enumerate(actions3[:10]):
            action_type = str(a.type).split('.')[-1] if hasattr(a.type, 'name') else str(a.type)
            print(f"  [{i}] type={action_type}, card_id={a.card_id}")
        
        # If in MAIN, check playability
        if gs.current_phase == dm_ai_module.Phase.MAIN:
            p0 = gs.players[0]
            print(f"\n=== MAIN_PHASE Debug ===")
            print(f"Hand: {len(p0.hand)} cards")
            print(f"Mana Zone: {len(p0.mana_zone)} cards (untapped: {sum(1 for c in p0.mana_zone if not c.is_tapped)})")
            
            # Manually check if cost-1 cards exist
            for card in p0.hand:
                card_def = card_db.get(card.card_id)
                if card_def:
                    cost = card_def.get('cost', 0)
                    if cost <= 1:
                        print(f"  PLAYABLE: card_id={card.card_id}, cost={cost}, name={card_def.get('name', 'Unknown')}")
    else:
        print("No PASS action found!")
