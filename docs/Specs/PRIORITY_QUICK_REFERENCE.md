# アクション優先度クイックリファレンス

## 優先度レベル（高→低）

### 🔴 Level 1: 必須応答（100） - 他すべてマスク
```
SELECT_TARGET     100  ← クエリ応答必須
SELECT_OPTION     100  ← クエリ応答必須
RESOLVE_EFFECT    100  ← 必須効果解決
```

### 🟠 Level 2: 高優先度（95-98） - フェーズアクションマスク
```
PAY_COST          98   ← スタック支払い
RESOLVE_PLAY      98   ← スタック解決
RESOLVE_EFFECT    95   ← オプショナル効果
```

### 🟡 Level 3: フェーズアクション（80-90）
```
MANA_CHARGE       90   ← MANA phase
ATTACK            85   ← ATTACK phase
BLOCK             85   ← BLOCK phase
DECLARE_PLAY      80   ← MAIN phase
```

### 🟢 Level 4: 低優先度（10-50）
```
SKIP_EFFECT       50   ← 効果スキップ
Other Actions     20   ← その他
Wrong Phase       10   ← 不適切フェーズ
```

### ⚪ Level 5: 最低（0）
```
PASS              0    ← 常に最後
```

---

## フェーズ別アクション表

| フェーズ | 許可アクション | マスクアクション |
|---------|----------------|------------------|
| **MANA** | MANA_CHARGE, PASS | DECLARE_PLAY, ATTACK, BLOCK |
| **MAIN** | DECLARE_PLAY, PASS | MANA_CHARGE, ATTACK, BLOCK |
| **ATTACK** | ATTACK_PLAYER/CREATURE, PASS | MANA_CHARGE, DECLARE_PLAY, BLOCK |
| **BLOCK** | BLOCK, PASS | MANA_CHARGE, DECLARE_PLAY, ATTACK |
| **START/DRAW/END** | (空) | すべて |

---

## 状態別マスキング

### 状態 A: クエリ待ち
```
✅ SELECT_TARGET, SELECT_OPTION
❌ すべての他アクション
```

### 状態 B: エフェクト待ち
```
✅ RESOLVE_EFFECT, SKIP_EFFECT
❌ フェーズアクション全て
```

### 状態 C: スタック処理中
```
✅ PAY_COST, RESOLVE_PLAY
❌ フェーズアクション全て
```

### 状態 D: 通常フェーズ
```
✅ フェーズ固有アクション + PASS
❌ 他フェーズのアクション
```

---

## IntentGeneratorの優先順位

```
1. waiting_for_user_input  → Query応答
2. !pending_effects.empty() → Effect解決
3. !stack.empty()           → Stack処理
4. phase-specific           → フェーズアクション
5. default                  → 空リスト（自動進行）
```

---

## SimpleAI実装ガイド

### 現在の問題
```cpp
// ❌ フェーズ無視、固定優先度
int get_priority(Action action) {
    if (action.type == MANA_CHARGE) return 40;  // MANAフェーズでも40
}
```

### 改善版
```cpp
// ✅ フェーズ対応、状態考慮
int get_priority(Action action, GameState state) {
    // Level 1: 必須応答
    if (action.type == RESOLVE_EFFECT) {
        if (is_optional(action, state)) return 95;
        return 100;
    }
    
    // Level 3: フェーズ依存
    if (state.current_phase == MANA) {
        if (action.type == MANA_CHARGE) return 90;  // MANAフェーズで90
    }
    if (state.current_phase == MAIN) {
        if (action.type == MANA_CHARGE) return 10;  // MAINフェーズで10
    }
}
```

---

## よくある間違い

### ❌ 間違い 1: エフェクト待ち中にPASS
```
pending_effects.size() > 0
actions = [RESOLVE_EFFECT, PASS]
selected = PASS  // ❌ 間違い
```
**正解**: RESOLVE_EFFECTを優先度100で選択

### ❌ 間違い 2: MANAフェーズでDECLARE_PLAY
```
current_phase = MANA
actions = [MANA_CHARGE(40), DECLARE_PLAY(80), PASS(0)]
selected = DECLARE_PLAY  // ❌ 間違い（固定優先度の場合）
```
**正解**: MANA_CHARGEを優先度90で選択（フェーズ対応）

### ❌ 間違い 3: クエリ待ち中に他アクション
```
waiting_for_user_input = true
actions = [SELECT_TARGET(90), PLAY_CARD(80)]  // ❌ IntentGeneratorの問題
```
**正解**: IntentGeneratorがSELECT_TARGETのみ生成すべき

---

## デバッグチェックリスト

### エフェクト解決が後回しになる場合

1. ✅ `pending_effects.size() > 0` を確認
2. ✅ IntentGeneratorがRESOLVE_EFFECTを生成しているか確認
3. ✅ SimpleAIの優先度が100になっているか確認
4. ✅ 他のアクションが混在していないか確認

### 不適切なフェーズでアクションが実行される場合

1. ✅ IntentGeneratorのphase switchが正しいか確認
2. ✅ current_phaseの値を確認
3. ✅ SimpleAIがフェーズを考慮しているか確認

---

## 図解リンク

- [アクション優先度フロー](docs/action_priority_flow.md) - Mermaid図
- [優先度ガントチャート](docs/priority_gantt.md) - 視覚的優先度
- [状態遷移図](docs/action_state_machine.md) - ステートマシン

---

**クイックリファレンス v1.0**  
**作成日**: 2026年2月7日  
**詳細仕様**: [PHASE_ACTION_PRIORITY_SPEC.md](PHASE_ACTION_PRIORITY_SPEC.md)
