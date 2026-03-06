# ツインパクトカード処理修正レポート

## 問題の概要
ツインパクトカードをプレイする際、`is_spell_side`フラグが無視され、常にメイン側（クリーチャー側）の定義が使われていた。その結果：
- 呪文側を選択してもクリーチャー側のコストで支払われる
- 呪文側を選択してもクリーチャー側の効果が実行される

## 根本原因
[src/engine/systems/game_logic_system.cpp](src/engine/systems/game_logic_system.cpp)の3箇所で`is_spell_side`フラグが使用されていなかった：

1. **PLAY_CARD case (Lines 84-115)**: メイン定義のみ使用
2. **DECLARE_PLAY case (Lines 168-231)**: メイン定義のみ使用
3. **handle_resolve_play (Lines 765-850)**: メイン定義のみ使用

## 修正内容

### 1. PLAY_CARD case修正
```cpp
// 修正前
const auto& def = card_db.at(c->card_id);

// 修正後
const auto& base_def = card_db.at(c->card_id);
const auto& def = (action.is_spell_side && base_def.spell_side) ? *base_def.spell_side : base_def;
```

### 2. DECLARE_PLAY case修正
```cpp
// 修正前
const auto& def = card_db.at(c->card_id);

// 修正後  
const auto& base_def = card_db.at(c->card_id);
const auto& def = (action.is_spell_side && base_def.spell_side) ? *base_def.spell_side : base_def;
```

### 3. handle_resolve_playへのis_spell_side伝達
```cpp
// PLAY_CARD/DECLARE_PLAYでhandle_resolve_playを呼び出す際
nlohmann::json resolve_args;
resolve_args["card"] = iid;
resolve_args["is_spell_side"] = action.is_spell_side; // 追加
```

### 4. handle_resolve_playでの定義取得修正
```cpp
// 修正前
if (card_db.count(card->card_id)) {
    const auto& def = card_db.at(card->card_id);
    if (RestrictionSystem::instance().is_play_forbidden(...)) return;
}
if (!card_db.count(card->card_id)) return;
const auto& def = card_db.at(card->card_id);

// 修正後
if (!card_db.count(card->card_id)) return;

bool is_spell_side = exec.resolve_int(inst.args.value("is_spell_side", 0)) != 0;
const auto& base_def = card_db.at(card->card_id);
const auto& def = (is_spell_side && base_def.spell_side) ? *base_def.spell_side : base_def;

if (RestrictionSystem::instance().is_play_forbidden(...)) return;
```

## 処理フロー（修正後）

### ツインパクトカードプレイ時
1. **phase_strategies.cpp**: 2つのDECLARE_PLAYアクションを生成
   - クリーチャー側: `is_spell_side = false`
   - 呪文側: `is_spell_side = true`

2. **dispatch_action (DECLARE_PLAY case)**:
   - `action.is_spell_side`を確認
   - Trueなら`base_def.spell_side`を使用
   - コスト支払い: 正しい側のコストで`auto_tap_mana()`

3. **handle_resolve_play**:
   - `inst.args["is_spell_side"]`を確認
   - Trueなら`base_def.spell_side`を使用
   - 効果処理: 正しい側の効果を実行

## カードデータ構造例
```json
{
  "id": 4,
  "name": "轟く革命 レッドギラゾーン",
  "type": "CREATURE",
  "cost": 4,
  "power": 6000,
  "effects": [/* クリーチャー効果 */],
  "spell_side": {
    "id": 4,
    "name": "♪俺の歌 聞けよ聞かなきゃ 殴り合い",
    "type": "SPELL",
    "cost": 3,
    "power": 0,
    "effects": [/* 呪文効果 */]
  }
}
```

## GUI側の対応状況

### 現在の実装
- **input_handler.py on_card_clicked()**: 複数のコマンドがある場合はダイアログで選択可能（すでに実装済み）
- **input_handler.py on_card_double_clicked()**: 最初に見つかったPLAY_CARDを実行（ツインパクト未対応）

### 改善提案（今後の実装）
```python
def on_card_double_clicked(self, card_id: int, instance_id: int):
    # ... 既存のコード ...
    
    # ツインパクト対応: 複数のPLAY_CARDがある場合はダイアログ表示
    play_cmds = [cmd for cmd, d in relevant_cmds if d.get('type') == 'PLAY_CARD']
    
    if len(play_cmds) > 1:
        # ツインパクト: ダイアログで選択
        items = []
        for cmd in play_cmds:
            desc = describe_command(cmd, self.gs, self.card_db)
            items.append({'description': desc, 'command': cmd})
        
        options = [item['description'] for item in items]
        item, ok = QInputDialog.getItem(
            self.window, "ツインパクト選択", "実行する面を選択してください:", options, 0, False
        )
        if ok and item:
            idx = options.index(item)
            self.session.execute_action(items[idx]['command'])
        return
    
    # 単一のPLAY_CARDなら通常通り実行
    if play_cmds:
        self.session.execute_action(play_cmds[0])
```

## SimpleAI対応状況
- **phase_strategies.cpp**: ツインパクトの両面を正しく生成
- **simple_ai.cpp get_priority()**: DECLARE_PLAYの優先度は80で共通（面による区別なし）

将来的な改善：spell_side優先度調整
```cpp
int SimpleAI::get_priority(const Action& action, const GameState& game_state) const {
    if (action.type == PlayerIntent::DECLARE_PLAY) {
        if (action.is_spell_side) {
            // 呪文面は状況により優先度調整
            Phase current_phase = game_state.current_phase;
            if (current_phase == Phase::MAIN && game_state.pending_effects.empty()) {
                return 85; // 効果解決がない時は呪文優先
            }
        }
        return 80;
    }
    // ...
}
```

## テスト方法

### 1. C++ユニットテスト
```cpp
TEST(TwinpactTest, SpellSideCostPayment) {
    GameState gs(42);
    gs.setup_test_duel();
    
    // ツインパクトカード（クリーチャー:4, 呪文:3）をP0の手札に追加
    CardInstance twinpact_card;
    twinpact_card.card_id = 4; // 轟く革命 レッドギラゾーン
    twinpact_card.instance_id = 100;
    twinpact_card.owner = 0;
    gs.players[0].hand.push_back(twinpact_card);
    
    // マナを3枚だけセット
    for (int i = 0; i < 3; i++) {
        CardInstance mana;
        mana.card_id = 1;
        mana.instance_id = 200 + i;
        gs.players[0].mana_zone.push_back(mana);
    }
    
    // 呪文側プレイ（コスト3）は成功するはず
    Action spell_play;
    spell_play.type = PlayerIntent::DECLARE_PLAY;
    spell_play.source_instance_id = 100;
    spell_play.is_spell_side = true;
    
    // 実行
    GameInstance gi(gs);
    gi.resolve_action(spell_play);
    
    // 検証: 呪文が墓地に、マナが3枚タップされているはず
    EXPECT_EQ(gs.players[0].graveyard.size(), 1);
    EXPECT_EQ(count_tapped_mana(gs, 0), 3);
}

TEST(TwinpactTest, CreatureSideCostPayment) {
    // 同様にクリーチャー側（コスト4）をテスト
    // マナ3枚では失敗、マナ4枚では成功するはず
}
```

### 2. GUIテスト
1. ツインパクトカード（ID: 4）を手札に配置
2. カードをクリック → ダイアログで2つの選択肢が表示されることを確認
3. 呪文側を選択 → コスト3で支払い、呪文効果が発動
4. クリーチャー側を選択 → コスト4で支払い、バトルゾーンに出現

### 3. AIテスト
```python
import dm_ai_module as dm

gs = dm.GameState(42)
gs.setup_test_duel()
card_db = dm.JsonLoader.load_cards('data/cards.json')

# ツインパクトを手札に追加
# ... (手札・マナ配置)

# アクション生成
actions = dm.ActionGenerator.generate_legal_actions(gs, card_db)

# 検証: ツインパクトカードから2つのDECLARE_PLAYが生成されているか
twinpact_actions = [a for a in actions if a.type == dm.PlayerIntent.DECLARE_PLAY and a.source_instance_id == 100]
assert len(twinpact_actions) == 2
assert any(a.is_spell_side for a in twinpact_actions)
assert any(not a.is_spell_side for a in twinpact_actions)
```

## ビルド手順
```powershell
# クリーンビルド
.\force_rebuild.ps1

# または増分ビルド
cmake --build build-msvc --config Release --target dm_ai_module
```

## 影響範囲
- **修正ファイル**: `src/engine/systems/game_logic_system.cpp` (1ファイル)
- **影響システム**: カードプレイ処理、コスト支払い、効果解決
- **互換性**: 既存の非ツインパクトカードに影響なし（`is_spell_side = false`がデフォルト）

## 次のステップ
1. ✅ C++エンジン修正完了
2. ⏳ ビルド & テスト実行
3. ⏳ GUI input_handlerのon_card_double_clicked修正（オプション）
4. ⏳ SimpleAIの呪文側優先度調整（オプション）
