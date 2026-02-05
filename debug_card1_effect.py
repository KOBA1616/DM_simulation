"""カードID=1の効果処理デバッグ"""
import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print("=== カードID=1効果デバッグ ===\n")

# カードDB読み込み
card_db = dm.JsonLoader.load_cards('data/cards.json')

# カードID=1の定義確認
if 1 in card_db:
    card_def = card_db[1]
    print(f"カード名: {card_def.name}")
    print(f"タイプ: {card_def.type}")
    print(f"コスト: {card_def.cost}")
    print(f"エフェクト数: {len(card_def.effects)}")
    
    for i, eff in enumerate(card_def.effects):
        print(f"\nエフェクト {i}:")
        print(f"  トリガー: {eff.trigger}")
        print(f"  コマンド数: {len(eff.commands)}")
        for j, cmd in enumerate(eff.commands):
            print(f"  コマンド {j}: {cmd.type}")
            print(f"    amount: {cmd.amount}")
            print(f"    optional: {cmd.optional}")
            print(f"    up_to: {cmd.up_to if hasattr(cmd, 'up_to') else 'N/A'}")
            print(f"    input_value_key: {cmd.input_value_key}")
            print(f"    output_value_key: {cmd.output_value_key}")
            if cmd.type == "QUERY":
                print(f"    str_param: {cmd.str_param}")
            if cmd.type == "TRANSITION":
                print(f"    from_zone: {cmd.from_zone}")
                print(f"    to_zone: {cmd.to_zone}")

# ゲーム作成してpending_effectsを確認
game = dm.GameInstance(12345, card_db)

config = dm.ScenarioConfig()
config.my_hand_cards = [1]
config.my_mana_zone = [1, 1]  # 水文明マナ2枚
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN

print("\n=== カードプレイ前 ===")
print(f"pending_effects: {len(game.state.pending_effects)}")

# カードをプレイ
p0 = game.state.players[0]
if len(p0.hand) > 0:
    card = p0.hand[0]
    
    actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
    declare_play_actions = [a for a in actions if int(a.type) == 15]
    
    if declare_play_actions:
        print(f"\nDECLARE_PLAYアクション実行...")
        game.resolve_action(declare_play_actions[0])
        
        print(f"\n=== カードプレイ後 ===")
        print(f"pending_effects数: {len(game.state.pending_effects)}")
        
        for i, pe in enumerate(game.state.pending_effects):
            print(f"\nPendingEffect {i}:")
            if isinstance(pe, dict):
                print(f"  (dict type): {pe}")
            else:
                print(f"  type: {pe.type}")
                print(f"  source_instance_id: {pe.source_instance_id}")
                print(f"  controller: {pe.controller}")
                print(f"  num_targets_needed: {pe.num_targets_needed}")
                print(f"  target_instance_ids: {pe.target_instance_ids}")
                print(f"  resolve_type: {pe.resolve_type}")
                print(f"  optional: {pe.optional}")
                print(f"  execution_context keys: {list(pe.execution_context.keys()) if hasattr(pe, 'execution_context') else 'N/A'}")
                if hasattr(pe, 'execution_context'):
                    for key, val in pe.execution_context.items():
                        print(f"    {key}: {val}")
                
                if hasattr(pe, 'effect_def') and pe.effect_def:
                    print(f"  effect_def:")
                    print(f"    actions count: {len(pe.effect_def.actions)}")
                    for j, act in enumerate(pe.effect_def.actions):
                        print(f"    Action {j}: {act.type}")
        
        # RESOLVE_EFFECTを1回実行
        actions_after = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
        resolve_actions = [a for a in actions_after if int(a.type) == 10]
        
        print(f"\n=== RESOLVE_EFFECT実行前 ===")
        print(f"RESOLVE_EFFECTアクション数: {len(resolve_actions)}")
        if resolve_actions:
            print(f"  slot_index: {resolve_actions[0].slot_index}")
        
        if resolve_actions:
            print(f"\nRESOLVE_EFFECTを1回実行...")
            game.resolve_action(resolve_actions[0])
            
            print(f"\n=== RESOLVE_EFFECT実行後 ===")
            print(f"pending_effects数: {len(game.state.pending_effects)}")
            
            for i, pe in enumerate(game.state.pending_effects):
                print(f"\nPendingEffect {i}:")
                if isinstance(pe, dict):
                    print(f"  (dict): {pe}")
                else:
                    print(f"  type: {pe.type}")
                    print(f"  num_targets_needed: {pe.num_targets_needed}")
                    print(f"  target_instance_ids: {pe.target_instance_ids}")
                    print(f"  optional: {pe.optional}")
                    if hasattr(pe, 'execution_context'):
                        print(f"  execution_context: {pe.execution_context}")
            
            # もう一度アクション生成
            actions_after2 = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
            resolve_actions2 = [a for a in actions_after2 if int(a.type) == 10]
            pass_actions2 = [a for a in actions_after2 if int(a.type) == 0]
            
            print(f"\nRESOLVE_EFFECT後のアクション:")
            print(f"  Total: {len(actions_after2)}")
            print(f"  RESOLVE_EFFECT: {len(resolve_actions2)}")
            print(f"  PASS: {len(pass_actions2)}")
            
            if resolve_actions2:
                print(f"  RESOLVE_EFFECT slot_index: {resolve_actions2[0].slot_index}")

print("\n=== デバッグ完了 ===")
