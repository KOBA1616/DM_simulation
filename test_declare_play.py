"""Test DECLARE_PLAY Stack Lifecycle Fix"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== DECLARE_PLAY Stack Lifecycle Test ===\\n")

# Load card database
card_db = dm.JsonLoader.load_cards('data/cards.json')
print("1. Card database loaded")

# Create game
game = dm.GameInstance(12345, card_db)
print("2. Game instance created")

# Setup scenario
config = dm.ScenarioConfig()
config.my_hand_cards = [1, 1, 1]
config.my_mana_zone = [1, 1, 1, 1, 1]
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN
print(f"3. Scenario setup complete - Phase={int(game.state.current_phase)}")
print(f"   P0: Hand={len(game.state.players[0].hand)}, Mana={len(game.state.players[0].mana_zone)}, Battle={len(game.state.players[0].battle_zone)}")

# Generate actions
actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
print(f"\\n4. Generated {len(actions)} actions")

# Check for DECLARE_PLAY actions
declare_play_actions = [a for a in actions if int(a.type) == 15]  # DECLARE_PLAY = 15
pass_actions = [a for a in actions if int(a.type) == 0]  # PASS = 0
print(f"   DECLARE_PLAY actions: {len(declare_play_actions)}")
print(f"   PASS actions: {len(pass_actions)}")

if declare_play_actions:
    print("\\n5. Executing DECLARE_PLAY action...")
    
    # Before state
    player = game.state.players[0]
    before_hand = len(player.hand)
    before_battle = len(player.battle_zone)
    before_grave = len(player.graveyard)
    before_stack = len(player.stack)
    print(f"   Before: Hand={before_hand}, Battle={before_battle}, Grave={before_grave}, Stack={before_stack}")
    
    # Execute action
    action = declare_play_actions[0]
    game.resolve_action(action)
    
    # After state
    player = game.state.players[0]
    after_hand = len(player.hand)
    after_battle = len(player.battle_zone)
    after_grave = len(player.graveyard)
    after_stack = len(player.stack)
    print(f"   After:  Hand={after_hand}, Battle={after_battle}, Grave={after_grave}, Stack={after_stack}")
    
    # Verify results
    print("\\n6. Results:")
    print(f"   Hand change: {before_hand} -> {after_hand} ({after_hand - before_hand:+d})")
    print(f"   Battle change: {before_battle} -> {after_battle} ({after_battle - before_battle:+d})")
    print(f"   Grave change: {before_grave} -> {after_grave} ({after_grave - before_grave:+d})")
    print(f"   Stack change: {before_stack} -> {after_stack} ({after_stack - before_stack:+d})")
    
    # Success criteria
    success = True
    messages = []
    
    if after_stack == 0:
        messages.append("[OK] Card NOT stuck on stack (old bug FIXED)")
    else:
        messages.append(f"[FAIL] Card stuck on stack: {after_stack}")
        success = False
    
    if after_hand < before_hand:
        messages.append("[OK] Card left hand")
    else:
        messages.append("[FAIL] Card did not leave hand")
        success = False
    
    if after_battle > before_battle or after_grave > before_grave:
        messages.append("[OK] Card reached final zone")
        if after_battle > before_battle:
            messages.append("  -> Battle zone (creature)")
        if after_grave > before_grave:
            messages.append("  -> Graveyard (spell)")
    else:
        messages.append("[FAIL] Card did not reach final zone")
        success = False
    
    print("\\n7. Validation:")
    for msg in messages:
        print("   " + msg)
    
    if success:
        print("\\n*** SUCCESS! DECLARE_PLAY now completes full Stack Lifecycle! ***")
    else:
        print("\\n*** PARTIAL SUCCESS or ISSUES DETECTED ***")
else:
    print("\\n[ERROR] No DECLARE_PLAY actions available")

print("\\n=== Test Complete ===")
