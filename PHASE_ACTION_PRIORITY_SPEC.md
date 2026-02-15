# ゲームフェーズ別アクション優先度とマスキング設計

## 問題の明確化

### 現在の問題点

1. **効果解決の優先度が不十分**
   - `pending_effects`がある状態でも、他のアクションが選択される可能性
   - アクション選択（SELECT_TARGET）より効果解決（RESOLVE_EFFECT）を明確に優先したい

2. **フェーズ別のアクションマスキングが未定義**
   - 各フェーズで生成されるすべてのアクションの仕様が明確でない
   - マスクすべき（生成すべきでない）アクションの基準が不明確

---

## IntentGeneratorの現在の優先順位

### 生成順序（優先度高→低）

[src/engine/actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp)の実装:

```cpp
std::vector<Action> IntentGenerator::generate_legal_actions(...) {
    // 1. HIGHEST PRIORITY: User Input Query Response
    if (game_state.waiting_for_user_input) {
        // SELECT_TARGET, SELECT_OPTION
        return query_response_actions;
    }

    // 2. VERY HIGH PRIORITY: Pending Effects
    if (!game_state.pending_effects.empty()) {
        // RESOLVE_EFFECT, SKIP_EFFECT (optional effects only)
        return pending_effect_actions;
    }

    // 3. HIGH PRIORITY: Stack Actions (Atomic Play Flow)
    auto stack_actions = StackStrategy::generate();
    if (!stack_actions.empty()) {
        // PAY_COST, RESOLVE_PLAY
        return stack_actions;
    }

    // 4. MEDIUM PRIORITY: Phase-Specific Actions
    switch (current_phase) {
        case MANA:    return ManaPhaseStrategy::generate();   // MANA_CHARGE, PASS
        case MAIN:    return MainPhaseStrategy::generate();   // DECLARE_PLAY, PASS
        case ATTACK:  return AttackPhaseStrategy::generate(); // ATTACK_PLAYER/CREATURE, PASS
        case BLOCK:   return BlockPhaseStrategy::generate();  // BLOCK, PASS
        default:      return {};  // Auto-advance
    }
}
```

### 現在の優先度レベル

| 優先度 | 条件 | アクション種別 | 例 |
|--------|------|----------------|-----|
| **1** | waiting_for_user_input | クエリ応答 | SELECT_TARGET, SELECT_OPTION |
| **2** | !pending_effects.empty() | エフェクト解決 | RESOLVE_EFFECT, SKIP_EFFECT |
| **3** | !stack.empty() | スタック処理 | PAY_COST, RESOLVE_PLAY |
| **4** | phase-specific | フェーズアクション | MANA_CHARGE, DECLARE_PLAY, ATTACK, BLOCK |

---

## 各フェーズの詳細仕様

### Phase 1: START_OF_TURN

**目的**: ターン開始処理（自動）

**生成アクション**: なし（空リスト）

**マスクすべきアクション**: すべて

**理由**: プレイヤーアクション不可、fast_forwardで自動進行

**処理フロー**:
```
START_OF_TURN → generate_legal_actions() → return {} → fast_forward() → DRAW phase
```

---

### Phase 2: DRAW

**目的**: ドローフェーズ（自動）

**生成アクション**: なし（空リスト）

**マスクすべきアクション**: すべて

**理由**: プレイヤーアクション不可、fast_forwardで自動進行

**処理フロー**:
```
DRAW → generate_legal_actions() → return {} → fast_forward() → MANA phase
```

---

### Phase 3: MANA

**目的**: マナチャージ

**生成アクション**:
```cpp
// ManaPhaseStrategy::generate()
if (!mana_charged_this_turn) {
    for (card in hand) {
        MANA_CHARGE(card)
    }
    PASS
} else {
    PASS  // Already charged
}
```

**許可アクション**:
- ✅ `MANA_CHARGE` - 手札からマナゾーンへ（1ターンに1回まで）
- ✅ `PASS` - マナチャージをスキップ

**マスクすべきアクション**:
- ❌ `DECLARE_PLAY` - MAINフェーズで行うべき
- ❌ `ATTACK_*` - ATTACKフェーズで行うべき
- ❌ すべての戦闘関連アクション

**特別ルール**:
- `mana_charged_by_player[active_pid] == true` の場合、PASS のみ生成
- マナチャージ済みフラグはターン終了時にリセット

---

### Phase 4: MAIN

**目的**: カードプレイ、攻撃など

**生成アクション**:
```cpp
// MainPhaseStrategy::generate()
for (card in hand) {
    if (can_play(card)) {
        DECLARE_PLAY(card, is_spell_side=false)
        if (card.is_twinpact && can_play(spell_side)) {
            DECLARE_PLAY(card, is_spell_side=true)
        }
        if (has_active_cost_reduction) {
            for (units in 1..max_units) {
                DECLARE_PLAY(card, payment_units=units)
            }
        }
    }
}
if (!actions.empty()) PASS
```

**許可アクション**:
- ✅ `DECLARE_PLAY` - カードプレイ宣言
  - 通常プレイ（Creature側）
  - Twinpactスペル側プレイ
  - Active Cost Reductionプレイ（Hyper Energyなど）
- ✅ `PASS` - メインフェーズ終了

**マスクすべきアクション**:
- ❌ `MANA_CHARGE` - MANAフェーズで行うべき
- ❌ `ATTACK_*` - ATTACKフェーズで行うべき
- ❌ `BLOCK` - BLOCKフェーズで行うべき

**プレイ可能条件**:
1. コスト支払い可能: `ManaSystem::can_pay_cost() == true`
2. Spellロック無し: `PassiveEffectSystem::check_restriction(CANNOT_USE_SPELLS) == false`
3. Cost Lockロック無し: `PassiveEffectSystem::check_restriction(LOCK_SPELL_BY_COST) == false`

**特別ケース**:
- アクションが空の場合、PASSを追加せず空リストを返す → 自動進行

---

### Phase 5: ATTACK (ATTACK_DECLARE)

**目的**: 攻撃宣言

**生成アクション**:
```cpp
// AttackPhaseStrategy::generate()
for (creature in active_player.battle_zone) {
    if (can_attack_player(creature) && !CANNOT_ATTACK) {
        ATTACK_PLAYER(creature, target_player=opponent)
    }
    if (can_attack_creature(creature) && !CANNOT_ATTACK) {
        for (tapped_enemy in opponent.battle_zone if tapped) {
            if (!protected_by_just_diver(tapped_enemy)) {
                ATTACK_CREATURE(creature, target=tapped_enemy)
            }
        }
    }
}
if (!actions.empty()) PASS
```

**許可アクション**:
- ✅ `ATTACK_PLAYER` - プレイヤー攻撃
  - 条件: `TargetUtils::can_attack_player() == true`
  - 条件: `!PassiveEffectSystem::check_restriction(CANNOT_ATTACK)`
- ✅ `ATTACK_CREATURE` - タップされたクリーチャー攻撃
  - 条件: `TargetUtils::can_attack_creature() == true`
  - 条件: ターゲットがタップ状態
  - 条件: Just Diver保護なし
- ✅ `PASS` - 攻撃しない

**マスクすべきアクション**:
- ❌ `MANA_CHARGE` - MANAフェーズで行うべき
- ❌ `DECLARE_PLAY` - MAINフェーズで行うべき
- ❌ `BLOCK` - BLOCKフェーズで行うべき
- ❌ アンタップクリーチャーへの攻撃（ルール違反）

**攻撃可能条件**:
1. クリーチャーがアンタップ
2. 召喚酔いなし（`turn_played < current_turn`）
3. `CANNOT_ATTACK`制約なし
4. Speed AttackerまたはTurn 2以降

**特別ルール**:
- Just Diver保護: `turn_played == current_turn` のクリーチャーは保護
- 攻撃不可能な場合、空リスト → 自動進行

---

### Phase 6: BLOCK (BLOCK_DECLARE)

**目的**: ブロッカー宣言

**生成アクション**:
```cpp
// BlockPhaseStrategy::generate()
for (blocker in defender.battle_zone) {
    if (!blocker.is_tapped && has_keyword(blocker, "BLOCKER")) {
        if (!CANNOT_BLOCK) {
            BLOCK(blocker)
        }
    }
}
if (!actions.empty()) PASS
```

**許可アクション**:
- ✅ `BLOCK` (PlayerIntent::BLOCK) - ブロッカー宣言
  - 条件: クリーチャーがアンタップ
  - 条件: BLOCKERキーワード所持
  - 条件: `!PassiveEffectSystem::check_restriction(CANNOT_BLOCK)`
- ✅ `PASS` - ブロックしない

**マスクすべきアクション**:
- ❌ `MANA_CHARGE` - MANAフェーズで行うべき
- ❌ `DECLARE_PLAY` - MAINフェーズで行うべき
- ❌ `ATTACK_*` - ATTACKフェーズで行うべき
- ❌ タップされたクリーチャーのブロック
- ❌ BLOCKERキーワードなしのブロック

**ブロック可能条件**:
1. ディフェンダー側のクリーチャー
2. アンタップ状態
3. BLOCKERキーワード所持
4. `CANNOT_BLOCK`制約なし

**特別ルール**:
- ブロック可能なクリーチャーがいない場合、空リスト → 自動進行

---

### Phase 7: END_OF_TURN

**目的**: ターン終了処理（自動）

**生成アクション**: なし（空リスト）

**マスクすべきアクション**: すべて

**理由**: プレイヤーアクション不可、fast_forwardで自動進行

---

## 特殊状態の優先度設計

### 状態 A: クエリ応答待ち (waiting_for_user_input)

**優先度**: **最高（Level 1）**

**発生タイミング**:
- カードプレイ時のターゲット選択
- カード効果のオプション選択

**生成アクション**:
```cpp
if (query_type == "SELECT_TARGET") {
    for (target_id in valid_targets) {
        SELECT_TARGET(target_id)
    }
}
if (query_type == "SELECT_OPTION") {
    for (i in 0..options.size()) {
        SELECT_OPTION(option_index=i)
    }
}
```

**許可アクション**:
- ✅ `SELECT_TARGET` - ターゲット選択
- ✅ `SELECT_OPTION` - オプション選択

**マスクすべきアクション**:
- ❌ **すべての通常アクション**（MANA_CHARGE, DECLARE_PLAY, ATTACK, etc.）

**重要**: クエリ応答完了まで、他のアクションは一切生成されない

---

### 状態 B: エフェクト解決待ち (pending_effects)

**優先度**: **非常に高（Level 2）**

**発生タイミング**:
- カードプレイ後のON_PLAY効果
- 攻撃後のON_ATTACK効果
- シールドブレイク時のSHIELD_TRIGGER効果
- バトル解決（RESOLVE_BATTLE）
- その他トリガー効果

**生成アクション**:
```cpp
// PendingEffectStrategy::generate()
for (effect in pending_effects) {
    if (effect.controller == decision_maker) {
        RESOLVE_EFFECT(effect_index)
        if (effect.optional) {
            SKIP_EFFECT(effect_index)
        }
    }
}
```

**許可アクション**:
- ✅ `RESOLVE_EFFECT` - エフェクト解決実行
- ✅ `SKIP_EFFECT` - オプショナル効果のスキップ（`optional=true`の場合のみ）

**マスクすべきアクション**:
- ❌ **すべてのフェーズアクション**（MANA_CHARGE, DECLARE_PLAY, ATTACK, BLOCK, PASS）

**エフェクト優先度（within pending_effects）**:
```
SHIELD_TRIGGER     = 1000  (最優先)
BREAK_SHIELD       = 800
RESOLVE_BATTLE     = 600
INTERNAL_PLAY      = 400
TRIGGER_ABILITY    = 200
Other              = 0
```

**スペル優先ルール**:
- スペル由来のエフェクトが存在する場合、スペルエフェクトのみ生成
- 非スペルエフェクトはスペル解決後に処理

**重要**: `pending_effects.empty() == false` の間、フェーズアクションは一切生成されない

---

### 状態 C: スタック処理中 (stack zone)

**優先度**: **高（Level 3）**

**発生タイミング**:
- DECLARE_PLAYアクション実行後
- カードがスタックゾーンに配置された状態

**生成アクション**:
```cpp
// StackStrategy::generate()
for (card in active_player.stack) {
    if (card.is_tapped) {
        RESOLVE_PLAY(card)  // コスト支払い済み
    } else if (can_pay_cost(card)) {
        PAY_COST(card)      // コスト支払い可能
    }
}
```

**許可アクション**:
- ✅ `PAY_COST` - マナコスト支払い
- ✅ `RESOLVE_PLAY` - プレイ解決実行

**マスクすべきアクション**:
- ❌ **すべてのフェーズアクション**（MANA_CHARGE, DECLARE_PLAY, ATTACK, BLOCK, PASS）

**重要**: スタックが空になるまで、フェーズアクションは一切生成されない

---

## 効果解決の明確な優先度設計

### 提案: 3段階優先度システム

#### Level 1: 必須応答（最優先）

**条件**: `waiting_for_user_input == true` OR `pending_effects[].optional == false`

**アクション**:
- `SELECT_TARGET` - ターゲット選択必須
- `SELECT_OPTION` - オプション選択必須
- `RESOLVE_EFFECT` - 非オプショナル効果解決必須

**SimpleAI優先度**: **100**（変更なし）

**マスキング**: **他のすべてのアクション**

---

#### Level 2: オプショナル効果（高優先度）

**条件**: `pending_effects[].optional == true`

**アクション**:
- `RESOLVE_EFFECT` - オプショナル効果解決
- `SKIP_EFFECT` - 効果スキップ

**SimpleAI優先度**: **95**（新規）

**マスキング**: **フェーズアクション** (MANA_CHARGE, DECLARE_PLAY, ATTACK, BLOCK, PASS)

---

#### Level 3: フェーズアクション（通常優先度）

**条件**: `pending_effects.empty() && stack.empty() && !waiting_for_user_input`

**アクション**:
- `MANA_CHARGE`, `DECLARE_PLAY`, `ATTACK_*`, `BLOCK`, `PASS`

**SimpleAI優先度**: **フェーズ依存**（Phase 1.1で実装予定）

**マスキング**: なし（IntentGeneratorで適切に生成）

---

## SimpleAI改善案（Phase 1.1）

### 現在の問題

[src/engine/ai/simple_ai.cpp](src/engine/ai/simple_ai.cpp):
```cpp
int SimpleAI::get_priority(const Action& action) {
    switch (action.type) {
        case RESOLVE_EFFECT: return 100;  // 最優先
        case SELECT_TARGET:  return 90;
        case PLAY_CARD:      return 80;
        case ATTACK:         return 60;
        case MANA_CHARGE:    return 40;   // ❌ MANAフェーズで低すぎ
        case PASS:           return 0;
    }
}
```

**問題点**:
1. フェーズ無視の固定優先度
2. オプショナル効果と必須効果の区別なし
3. MANAフェーズでMANA_CHARGEが低優先度

---

### 改善実装（フェーズ対応版）

```cpp
int SimpleAI::get_priority(const Action& action, const GameState& state) {
    // Level 1: 必須応答（最優先）
    if (action.type == PlayerIntent::RESOLVE_EFFECT) {
        // Check if effect is optional
        if (action.slot_index >= 0 && action.slot_index < state.pending_effects.size()) {
            const auto& effect = state.pending_effects[action.slot_index];
            if (effect.optional) {
                return 95;  // Level 2: オプショナル効果
            }
        }
        return 100;  // Level 1: 必須効果解決
    }
    
    if (action.type == PlayerIntent::SELECT_TARGET || 
        action.type == PlayerIntent::SELECT_OPTION) {
        return 100;  // Level 1: クエリ応答必須
    }
    
    // Level 2: スタック処理
    if (action.type == PlayerIntent::PAY_COST || 
        action.type == PlayerIntent::RESOLVE_PLAY) {
        return 98;  // Stack処理は高優先度
    }
    
    // Level 2: オプショナル効果スキップ
    if (action.type == PlayerIntent::SKIP_EFFECT) {
        return 50;  // スキップは低めに設定（解決を優先）
    }
    
    // PASS is always lowest
    if (action.type == PlayerIntent::PASS) {
        return 0;
    }
    
    // Level 3: フェーズ別優先度
    switch (state.current_phase) {
        case GamePhase::MANA:
            if (action.type == PlayerIntent::MANA_CHARGE) return 90;
            return 10;
        
        case GamePhase::MAIN:
            if (action.type == PlayerIntent::DECLARE_PLAY) return 80;
            return 20;
        
        case GamePhase::ATTACK:
            if (action.type == PlayerIntent::ATTACK_PLAYER ||
                action.type == PlayerIntent::ATTACK_CREATURE) return 85;
            return 10;
        
        case GamePhase::BLOCK:
            if (action.type == PlayerIntent::BLOCK) return 85;
            return 10;
        
        default:
            return 20;
    }
}
```

---

## 優先度まとめ表

### 完全優先度マトリクス

| アクション | 状態/フェーズ | 優先度 | 説明 |
|-----------|--------------|--------|------|
| **SELECT_TARGET** | Query待ち | 100 | 必須応答 |
| **SELECT_OPTION** | Query待ち | 100 | 必須応答 |
| **RESOLVE_EFFECT (必須)** | pending_effects | 100 | 必須効果解決 |
| **PAY_COST** | Stack処理 | 98 | スタック支払い |
| **RESOLVE_PLAY** | Stack処理 | 98 | スタック解決 |
| **RESOLVE_EFFECT (optional)** | pending_effects | 95 | オプショナル効果 |
| **MANA_CHARGE** | MANA phase | 90 | フェーズ主目的 |
| **ATTACK** | ATTACK phase | 85 | フェーズ主目的 |
| **BLOCK** | BLOCK phase | 85 | フェーズ主目的 |
| **DECLARE_PLAY** | MAIN phase | 80 | フェーズ主目的 |
| **SKIP_EFFECT** | pending_effects | 50 | 効果スキップ |
| **Other** | Any | 20 | その他アクション |
| **MANA_CHARGE** | MAIN/ATTACK/BLOCK | 10 | 不適切フェーズ |
| **DECLARE_PLAY** | MANA/ATTACK/BLOCK | 10 | 不適切フェーズ |
| **ATTACK** | MANA/MAIN/BLOCK | 10 | 不適切フェーズ |
| **PASS** | Any | 0 | 最低優先度 |

---

## 実装推奨事項

### 即座に実装すべき改善

1. **SimpleAI::get_priority() のフェーズ対応化**
   - `get_priority(action, state)` シグネチャに変更
   - フェーズ別優先度マトリクス適用
   - オプショナル効果の区別

2. **デバッグログ強化**
   ```cpp
   std::cout << "[SimpleAI] Phase=" << static_cast<int>(state.current_phase)
             << " ActionType=" << static_cast<int>(action.type)
             << " Priority=" << priority
             << " pending_effects=" << state.pending_effects.size()
             << " waiting_input=" << state.waiting_for_user_input << "\n";
   ```

3. **テストケース追加**
   - `test_priority_with_pending_effects.py`
   - `test_priority_phase_aware.py`
   - `test_priority_optional_effects.py`

### 将来的な改善

1. **IntentGeneratorレベルでのマスキング強化**
   - 不適切なアクションを生成段階で除外
   - 現在はAI優先度で対処しているが、生成自体を制限すべき

2. **エフェクトチェーン優先度**
   - 複数のトリガーが同時発火する場合の処理順序
   - ターンプレイヤー優先ルールの実装

---

## まとめ

### ✅ 達成事項

1. **全フェーズのアクション仕様明確化**
   - MANA, MAIN, ATTACK, BLOCK各フェーズの詳細
   - 許可/禁止アクションの完全リスト

2. **優先度システムの3段階分類**
   - Level 1: 必須応答（100）
   - Level 2: 高優先度（95-98）
   - Level 3: フェーズアクション（10-90）

3. **効果解決の明確な優先度設定**
   - 必須効果: 100
   - オプショナル効果: 95
   - 通常アクションより常に優先

### 🚀 次のステップ

**Phase 1.1実装**: フェーズ対応SimpleAI
- 所要時間: 1-2時間
- [DESIGN_PHASE_AWARE_AI.cpp](native_prototypes/DESIGN_PHASE_AWARE_AI.cpp)のコードを参考に実装

---

**ドキュメント作成日**: 2026年2月7日  
**バージョン**: 1.0  
**参照実装**:
- [intent_generator.cpp](src/engine/actions/intent_generator.cpp)
- [phase_strategies.cpp](src/engine/actions/strategies/phase_strategies.cpp)
- [pending_strategy.cpp](src/engine/actions/strategies/pending_strategy.cpp)
- [stack_strategy.cpp](src/engine/actions/strategies/stack_strategy.cpp)
- [simple_ai.cpp](src/engine/ai/simple_ai.cpp)
