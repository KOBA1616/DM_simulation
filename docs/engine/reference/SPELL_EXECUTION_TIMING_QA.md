# 呪文効果実行タイミング - Q&A

## Q: 呪文効果実行時はスタックで発動・処理されるか？

**A: はい、呪文の効果はすべてスタック上で実行されます。**

## 詳細説明

### カードの位置と処理タイミング

```
時系列: プレイから完了まで

┌────────────────────────────────────────────┐
│ 1. プレイ宣言                              │
│    位置: HAND → STACK                      │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 2. コスト支払い                            │
│    位置: STACK (tapped)                    │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 3. 解決開始 (RESOLVE_PLAY)                 │
│    位置: STACK ⭐                          │
│    処理: handle_resolve_play() 呼び出し    │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 4. 効果コンパイル                          │
│    位置: STACK ⭐                          │
│    処理: 各効果を compiled_effects に追加  │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 5. 効果実行                                │
│    位置: STACK ⭐⭐⭐                      │
│    処理:                                   │
│    - ドロー効果                            │
│    - 破壊効果                              │
│    - その他すべての効果                    │
│                                            │
│    ※この段階でカードはまだスタック上      │
└────────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────────┐
│ 6. ゾーン移動                              │
│    位置: STACK → GRAVEYARD (or 置換先)     │
│    処理: TransitionCommand 実行            │
└────────────────────────────────────────────┘
```

### コードによる証明

**エンジンコード**: `src/engine/systems/game_logic_system.cpp`

```cpp
void GameLogicSystem::handle_resolve_play(
    PipelineExecutor& exec, 
    GameState& state, 
    const Instruction& inst,
    const std::map<core::CardID, core::CardDefinition>& card_db
) {
    // ステップ1: カード取得（この時点でスタック上）
    int instance_id = exec.resolve_int(inst.args.value("card", 0));
    const CardInstance* card = state.get_card_instance(instance_id);
    if (!card) return;
    
    // ★証明: この時点で、カードはstate.players[X].stack に存在
    
    const auto& def = card_db.at(card->card_id);
    
    if (def.type == CardType::SPELL) {
        std::vector<Instruction> compiled_effects;
        std::map<std::string, int> ctx;
        
        // ステップ2: 効果をコンパイル（カードはまだスタック上）
        for (const auto& eff : def.effects) {
            // ★各効果は instance_id を使ってスタック上のカードを参照
            EffectSystem::instance().compile_effect(
                state, eff, instance_id, ctx, card_db, compiled_effects
            );
        }
        
        // ステップ3: 墓地移動を最後に追加
        nlohmann::json move_args;
        move_args["target"] = instance_id;
        move_args["to"] = "GRAVEYARD";
        compiled_effects.emplace_back(InstructionOp::MOVE, move_args);
        
        // ステップ4: パイプラインに登録
        auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
        exec.call_stack.push_back({block, 0, LoopContext{}});
    }
}

// パイプライン実行（resolve_play_from_stack から呼ばれる）
void GameLogicSystem::resolve_play_from_stack(...) {
    PipelineExecutor pipeline;
    
    // handle_resolve_play を呼び出し（効果をコンパイル）
    handle_resolve_play(pipeline, game_state, inst, card_db);
    
    // パイプライン実行（効果 → 移動の順）
    pipeline.execute(nullptr, game_state, card_db);
    // ★ここで初めて、効果が実行され、その後移動が実行される
}
```

## 重要な結論

### ✅ 確定事項

1. **効果実行中のカード位置**: スタック（STACK）
2. **効果実行のタイミング**: ゾーン移動の前
3. **ゾーン移動のタイミング**: すべての効果実行後

### 実行順序の保証

```
1. 効果1実行（カードはSTACK上）
2. 効果2実行（カードはSTACK上）
3. 効果3実行（カードはSTACK上）
   ...
N. すべての効果完了（カードはまだSTACK上）
N+1. ゾーン移動実行（STACK → GRAVEYARD）
```

### なぜこの仕様なのか？

1. **参照の一貫性**: 効果実行中、カードの `instance_id` で安定的に参照できる
2. **トリガーの正確性**: 効果実行後のゾーン移動で、正しいイベントが発火
3. **置換効果の実装**: 墓地への移動を検出・置換できる
4. **アンドゥ/リドゥ**: コマンドベースで正確な状態復元が可能

## 実例: カードを2枚引く呪文

```json
{
  "type": "SPELL",
  "effects": [
    {
      "type": "DRAW_CARD",
      "amount": 2
    }
  ]
}
```

### 実行フロー

```
T0: プレイ宣言
    → 手札からスタックへ移動
    
T1: コスト支払い
    → スタック上でタップ
    
T2: handle_resolve_play() 呼び出し
    カード位置: state.players[0].stack[0] ⭐
    
T3: DRAW_CARD 効果コンパイル
    カード位置: state.players[0].stack[0] ⭐
    
T4: GRAVEYARD 移動命令追加
    カード位置: state.players[0].stack[0] ⭐
    
T5: パイプライン実行開始
    
T6: DRAW_CARD 効果実行
    カード位置: state.players[0].stack[0] ⭐⭐⭐
    処理: カードを2枚引く
    
T7: TransitionCommand 実行
    処理: STACK から GRAVEYARD へ移動
    カード位置: state.players[0].stack から削除
    カード位置: state.players[0].graveyard に追加
    
T8: 完了
    カード位置: state.players[0].graveyard ✅
```

## 置換効果との関係

置換効果（REPLACE_CARD_MOVE）も、**スタック上で処理されます**：

```
T6: DRAW_CARD 効果実行（スタック上）
T7: REPLACE_CARD_MOVE 効果実行（スタック上）
    → 墓地への移動を検出
    → 移動先を変更
T8: TransitionCommand 実行
    → STACK → DECK_BOTTOM（墓地をスキップ）
```

## まとめ

| 質問 | 回答 |
|------|------|
| 呪文効果はスタックで処理されるか？ | **はい** ✅ |
| 効果実行中、カードはどこにあるか？ | **スタック** ⭐ |
| いつ墓地に移動するか？ | **すべての効果実行後** |
| 置換効果もスタック上で処理されるか？ | **はい** ✅ |

---

**参照ドキュメント**:
- [呪文のゾーン経路と置換効果（詳細版）](SPELL_ZONE_FLOW_AND_REPLACEMENT.md)
- [呪文の置換効果 - クイックリファレンス](SPELL_REPLACEMENT_QUICK_REF.md)

**エンジンコード**:
- `src/engine/systems/game_logic_system.cpp:handle_resolve_play()`
- `src/engine/systems/game_logic_system.cpp:resolve_play_from_stack()`
- `src/engine/game_command/commands.cpp:TransitionCommand::execute()`

**最終更新**: 2026年1月11日
