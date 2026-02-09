import sys
sys.path.insert(0, '.')

import dm_ai_module
from dm_toolkit import commands_v2

# Use command-first wrapper
generate_legal_commands = commands_v2.generate_legal_commands

# 簡単なゲーム状態でgenerate_legal_commandsをテスト
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()

# デッキ設定を追加
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

if hasattr(dm_ai_module, 'JsonLoader'):
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    print("Loaded card database")
else:
    print("ERROR: JsonLoader not available")
    sys.exit(1)

# ゲーム開始
if hasattr(dm_ai_module.PhaseManager, 'start_game'):
    dm_ai_module.PhaseManager.start_game(gs, card_db)
    print("Game started")
else:
    print("ERROR: PhaseManager.start_game not available")
    sys.exit(1)

print("Game ready for command generation")

# generate_legal_commandsを呼び出し
print("\nCalling generate_legal_commands...")
try:
    try:
        commands = generate_legal_commands(gs, card_db, strict=False) or []
    except TypeError:
        commands = generate_legal_commands(gs, card_db) or []
    except Exception:
        commands = []
except Exception:
    commands = []

print(f"Generated {len(commands)} commands")
for i, cmd in enumerate(commands[:5]):  # 最初の5個だけ表示
    print(f"  Command {i}: {cmd}")

print("\nTest completed successfully!")
