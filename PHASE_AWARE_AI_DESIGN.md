# フェーズ別優先度設計ドキュメント

## 問題提起

現在のSimpleAI実装は**フェーズに関係なく固定の優先度**を使用しています。

### 現在の問題点

```cpp
// 現在の実装（simple_ai.cpp）
int SimpleAI::get_priority(const Action& action) {
    switch (action.type) {
        case RESOLVE_EFFECT: return 100;
        case PLAY_CARD: return 80;
        case ATTACK: return 60;
        case MANA_CHARGE: return 40;  // ❌ MANAフェーズでも優先度40
        case PASS: return 0;
    }
}
```

**問題例**:
- **MANAフェーズ**: MANA_CHARGEが優先度40 → 本来90にすべき
- **ATTACK_DECLAREフェーズ**: ATTACKが優先度60 → 本来85にすべき
- **MAINフェーズ**: どのアクションも同じ優先度 → 場面に応じた調整が必要

---

## 解決策: フェーズ別優先度システム

### 優先度マトリクス

| アクション | MANA | MAIN | ATTACK_DECLARE | BLOCK_DECLARE | 
|-----------|------|------|----------------|---------------|
| RESOLVE_EFFECT | 100 | 100 | 100 | 100 |
| SELECT_TARGET | 90 | 90 | 90 | 90 |
| **MANA_CHARGE** | **90** ⬆️ | 10 | 10 | 10 |
| PLAY_CARD | 10 | **80** | 10 | 10 |
| **ATTACK** | 10 | 60 | **85** ⬆️ | 10 |
| **DECLARE_BLOCKER** | 10 | 10 | 10 | **85** ⬆️ |
| PASS | 0 | 0 | 0 | 0 |

### 設計原則

1. **RESOLVE_EFFECTは常に最優先** (100)
   - エフェクト解決は必須、中断不可

2. **SELECT_TARGET/SELECT_OPTIONは常に高優先** (90)
   - クエリレスポンスは即座に処理

3. **PASSは常に最低優先** (0)
   - すべてのアクションを試してから最後にパス

4. **フェーズごとの主要アクションを85-90に**
   - MANAフェーズ: MANA_CHARGE = 90
   - ATTACK_DECLAREフェーズ: ATTACK = 85
   - BLOCK_DECLAREフェーズ: DECLARE_BLOCKER = 85

5. **他フェーズでの不適切なアクションは10に**
   - MAINフェーズでのMANA_CHARGE = 10（低優先度）

---

## 実装方法

### 現在の実装（Phase 1）

```cpp
// simple_ai.cpp
int SimpleAI::get_priority(const Action& action) {
    // フェーズ不使用、固定優先度
    switch (action.type) {
        case RESOLVE_EFFECT: return 100;
        case PLAY_CARD: return 80;
        // ...
    }
}
```

### 改善実装（フェーズ対応版）

```cpp
// simple_ai.cpp (改善版)
int SimpleAI::get_priority(const Action& action, const GameState& state) {
    // Universal priorities
    if (action.type == PlayerIntent::RESOLVE_EFFECT) return 100;
    if (action.type == PlayerIntent::SELECT_TARGET) return 90;
    if (action.type == PlayerIntent::PASS) return 0;
    
    // Phase-specific priorities
    switch (state.current_phase) {
        case GamePhase::MANA:
            if (action.type == PlayerIntent::MANA_CHARGE) return 90;
            return 10;  // Other actions low priority
        
        case GamePhase::MAIN:
            if (action.type == PlayerIntent::PLAY_CARD) return 80;
            if (action.type == PlayerIntent::ATTACK) return 60;
            return 20;
        
        case GamePhase::ATTACK_DECLARE:
            if (action.type == PlayerIntent::ATTACK) return 85;
            return 10;
        
        case GamePhase::BLOCK_DECLARE:
            if (action.type == PlayerIntent::DECLARE_BLOCKER) return 85;
            if (action.type == PlayerIntent::DECLARE_NO_BLOCK) return 10;
            return 10;
        
        default:
            return 20;  // Fallback for other phases
    }
}
```

---

## 各フェーズの詳細解説

### 1. MANAフェーズ

**目的**: マナゾーンにカードをチャージ

**優先度**:
```
RESOLVE_EFFECT: 100  // エフェクト解決（必須）
MANA_CHARGE: 90      // マナチャージ（フェーズの主目的）
PASS: 0              // チャージしないならパス
```

**理由**: MANAフェーズの目的は明確（マナチャージ）なので、MANA_CHARGEを最優先にすべき。

---

### 2. MAINフェーズ

**目的**: カードプレイ、攻撃、能力使用など

**優先度**:
```
RESOLVE_EFFECT: 100   // エフェクト解決（必須）
SELECT_TARGET: 90     // ターゲット選択（クエリ応答）
PLAY_CARD: 80         // カードプレイ（主要アクション）
ATTACK: 60            // 攻撃（サブアクション）
Other abilities: 20   // その他の能力
MANA_CHARGE: 10       // 不適切（MANAフェーズで行うべき）
PASS: 0               // フェーズ終了
```

**理由**: 
- カードプレイが最優先（デッキを展開）
- 攻撃は次に重要（勝利条件）
- MANA_CHARGEはこのフェーズでは不適切なので低優先度

---

### 3. ATTACK_DECLAREフェーズ

**目的**: 攻撃クリーチャーを宣言

**優先度**:
```
RESOLVE_EFFECT: 100   // エフェクト解決（必須）
SELECT_TARGET: 90     // ターゲット選択
ATTACK: 85            // 攻撃宣言（フェーズの主目的）
Other: 10             // その他のアクション
PASS: 0               // 攻撃しない
```

**理由**: このフェーズの目的は攻撃宣言なので、ATTACKを85に引き上げる。

---

### 4. BLOCK_DECLAREフェーズ

**目的**: ブロッククリーチャーを宣言

**優先度**:
```
RESOLVE_EFFECT: 100      // エフェクト解決（必須）
SELECT_TARGET: 90        // ターゲット選択
DECLARE_BLOCKER: 85      // ブロック宣言（フェーズの主目的）
DECLARE_NO_BLOCK: 10     // ブロックしない（低優先度）
Other: 10                // その他のアクション
PASS: 0                  // フェーズ終了
```

**理由**: 
- ブロック可能ならブロックすべき（DECLARE_BLOCKER: 85）
- ブロックしない選択は低優先度（DECLARE_NO_BLOCK: 10）

---

## テスト計画

### テストケース1: MANAフェーズでのMANA_CHARGE優先

```python
# TestCase: MANA phase prioritizes MANA_CHARGE
gs.current_phase = GamePhase.MANA
actions = [
    Action(PLAY_CARD, ...),      # 優先度10
    Action(MANA_CHARGE, ...),    # 優先度90 ← これが選ばれる
    Action(PASS, ...)            # 優先度0
]
selected = SimpleAI.select_action(actions, gs)
assert actions[selected].type == MANA_CHARGE
```

### テストケース2: ATTACK_DECLAREフェーズでのATTACK優先

```python
# TestCase: ATTACK_DECLARE phase prioritizes ATTACK
gs.current_phase = GamePhase.ATTACK_DECLARE
actions = [
    Action(PLAY_CARD, ...),      # 優先度10
    Action(ATTACK, ...),         # 優先度85 ← これが選ばれる
    Action(PASS, ...)            # 優先度0
]
selected = SimpleAI.select_action(actions, gs)
assert actions[selected].type == ATTACK
```

### テストケース3: BLOCK_DECLAREフェーズでのDECLARE_BLOCKER優先

```python
# TestCase: BLOCK_DECLARE phase prioritizes DECLARE_BLOCKER
gs.current_phase = GamePhase.BLOCK_DECLARE
actions = [
    Action(DECLARE_NO_BLOCK, ...),  # 優先度10
    Action(DECLARE_BLOCKER, ...),   # 優先度85 ← これが選ばれる
    Action(PASS, ...)               # 優先度0
]
selected = SimpleAI.select_action(actions, gs)
assert actions[selected].type == DECLARE_BLOCKER
```

---

## 実装スケジュール

### Phase 1.1: フェーズ対応AI実装（1-2時間）

1. **simple_ai.hpp更新**
   - `get_priority(action, state)`にシグネチャ変更
   - ドキュメント更新

2. **simple_ai.cpp更新**
   - フェーズ別優先度ロジック実装
   - デバッグログにフェーズ情報追加

3. **CMakeLists.txt**
   - 変更不要（ファイル追加なし）

4. **テスト作成**
   - `test_phase_aware_ai.py`作成
   - 各フェーズでの優先度テスト

### Phase 1.2: ビルドとテスト（30分）

1. ビルド
2. テスト実行
3. レポート作成（PHASE1_1_REPORT.md）

---

## 期待される効果

### Before（現在）
```
MANAフェーズ:
  Actions: [MANA_CHARGE(40), PLAY_CARD(80), PASS(0)]
  → PLAY_CARD選択 ❌ (不適切)

ATTACK_DECLAREフェーズ:
  Actions: [ATTACK(60), PLAY_CARD(80), PASS(0)]
  → PLAY_CARD選択 ❌ (不適切)
```

### After（フェーズ対応後）
```
MANAフェーズ:
  Actions: [MANA_CHARGE(90), PLAY_CARD(10), PASS(0)]
  → MANA_CHARGE選択 ✅

ATTACK_DECLAREフェーズ:
  Actions: [ATTACK(85), PLAY_CARD(10), PASS(0)]
  → ATTACK選択 ✅
```

### パフォーマンス影響
- **計算コスト**: 無視できる（フェーズ判定はO(1)）
- **コード複雑度**: 中程度増加（switch文が2段階に）
- **保守性**: 向上（フェーズごとの意図が明確）

---

## まとめ

### 推奨事項

✅ **フェーズ別優先度を実装すべき**

**理由**:
1. ゲームルールとの整合性（各フェーズの目的を反映）
2. AI行動の予測可能性向上
3. デバッグ容易性（フェーズごとの挙動が明確）

### 実装方法

**Option A: 完全なフェーズ対応**（推奨）
- すべてのフェーズで最適な優先度を設定
- 実装コスト: 中（1-2時間）
- メリット: 最も正確な動作

**Option B: 部分的なフェーズ対応**
- 重要フェーズ（MANA, ATTACK_DECLARE, BLOCK_DECLARE）のみ対応
- 実装コスト: 低（30-60分）
- メリット: 早期リリース可能

**Option C: 現状維持**
- フェーズ非対応のまま
- 実装コスト: なし
- デメリット: 不適切なアクション選択が発生

### 次のステップ

Phase 1.1として、フェーズ対応AIを実装することを推奨します。

```powershell
# 実装手順
1. simple_ai.hpp/cpp更新
2. test_phase_aware_ai.py作成
3. ビルド・テスト
4. Phase 1完了から Phase 1.1完了へ更新
```

---

**ドキュメント作成日**: 2026年2月7日  
**提案者**: AI Assistant  
**ステータス**: 設計案（実装待ち）
