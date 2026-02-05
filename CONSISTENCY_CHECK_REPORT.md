# 整合性チェックレポート

**日付**: 2026年2月5日  
**対象**: マナチャージ重複実行とカード実行スキップの修正

## 実施した整合性チェック

### 1. プログラムカウンタ（pc）の二重インクリメント問題

#### 検証内容
- `exec.call_stack[parent_idx].pc++`の手動インクリメントを全てチェック
- `pipeline_executor.cpp`の自動pc++ロジックとの競合を確認

#### 結果
✅ **整合性OK**

- `game_logic_system.cpp`内の3箇所の手動pc++は**正当**:
  - 行124, 130, 394（`dispatch_action`内）
  - これらは一時的な`pipeline`変数を使用し、後で`execute()`が呼ばれる前の準備段階
  - `pipeline_executor.cpp`の自動pc++とは別のコンテキストで動作

- `handle_*`関数群内の手動pc++は**すべて削除済み**（5箇所）:
  - 行545, 608, 681, 791, 942（削除済み）
  - これらは`exec`参照を使用し、既にexecuteループ内で実行中のため、自動pc++と競合していた

### 2. マナチャージフラグチェックの網羅性

#### 検証内容
- `mana_charged_by_player`フラグの使用箇所を全て確認
- マナチャージの実行経路を追跡し、チェック漏れを検証

#### 発見した問題
❌ **整合性問題発見**: `handle_mana_charge`にフラグチェックがなかった

#### 実施した修正
✅ **修正完了**: [game_logic_system.cpp:1096-1128](game_logic_system.cpp#L1096-L1128)

```cpp
void GameLogicSystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
    // 1. カードIDの取得と検証
    int card_id = exec.resolve_int(inst.args.value("card", 0));
    if (card_id < 0) return;

    // 2. DM Rule: マナチャージ済みフラグのチェック（追加）
    const CardInstance* card_ptr = state.get_card_instance(card_id);
    if (!card_ptr) return;
    
    PlayerID owner = card_ptr->owner;
    if (state.turn_stats.mana_charged_by_player[owner]) {
        // 既にチャージ済み - 早期リターン
        return;
    }

    // 3. マナチャージ実行
    Instruction move = utils::ActionPrimitiveUtils::create_mana_charge_instruction(card_id);
    auto block = std::make_shared<std::vector<Instruction>>();
    block->push_back(move);
    exec.call_stack.push_back({block, 0, LoopContext{}});
    
    // 4. フラグ設定（undo対応）
    auto flow_cmd = std::make_shared<game_command::FlowCommand>(
        game_command::FlowCommand::FlowType::SET_MANA_CHARGED, 1);
    state.execute_command(std::move(flow_cmd));
}
```

#### マナチャージの実行経路

1. **ManaChargeCommand経由**（C++コマンド）
   - ✅ フラグチェックあり（[action_commands.cpp:114](action_commands.cpp#L114)）
   
2. **PlayerIntent::MANA_CHARGE経由**（UI/AI）
   - `dispatch_action` → `handle_mana_charge`
   - ✅ 今回の修正でフラグチェック追加

3. **GAME_ACTION命令経由**（パイプライン）
   - `pipeline_executor` → `handle_mana_charge`
   - ✅ 今回の修正でフラグチェック追加

### 3. その他の整合性確認

#### call_stack.push_back()の使用箇所
検証した10箇所すべてで適切な処理を確認:
- 行128, 392: `dispatch_action`内（手動pc++あり、正当）
- 行541, 601, 671, 778, 935, 1034, 1092, 1104: `handle_*`関数内（手動pc++なし、正しい）

#### TurnStatsのリセット
✅ 適切に実装されている:
- `FlowCommand::FlowType::RESET_TURN_STATS`でリセット
- undo対応も完備

## コンパイル結果

✅ **ビルド成功**
- エラー: 0件
- 警告: 型変換警告のみ（既存の問題、機能に影響なし）

## まとめ

### 修正内容
1. **カード実行スキップ問題**: `handle_*`関数群内の手動pc++を5箇所削除 ✅
2. **マナチャージ重複実行問題**:
   - `ManaChargeCommand::execute()`にフラグチェック追加 ✅
   - `handle_mana_charge()`にフラグチェック追加 ✅

### 整合性評価
- **プログラムカウンタ管理**: ✅ 整合性確認済み
- **マナチャージフラグ**: ✅ 全経路で保護済み
- **コマンドパターン**: ✅ undo対応完備
- **ビルド**: ✅ エラーなし

### 推奨事項
今後の開発で同様の問題を防ぐため：
1. `exec.call_stack.push_back()`後の手動pc++は原則禁止
2. ゲームルール制約（1ターン1回など）は全実行経路でチェック
3. 新しいコマンド追加時は整合性チェックリストを参照
