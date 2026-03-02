# PlayerIntent アクション型完全リファレンス

## 定義場所
[src/core/action.hpp](src/core/action.hpp)

---

## 全アクション一覧（22種類）

### 📋 カテゴリ分類

```
ユーザーアクション (15種類): プレイヤーが直接選択できるアクション
エンジン内部アクション (7種類): システムが自動生成するアクション
```

---

## ✅ ユーザーアクション（User Actions）

### 1. PASS
**用途**: フェーズ・ステップの終了  
**生成フェーズ**: すべてのフェーズ（アクションがある場合）  
**パラメータ**: なし  
**説明**: 
- 現在のフェーズで何もアクションを実行せず次に進む
- 通常、各フェーズで最低優先度（SimpleAI優先度: 0）
- すべてのアクションリストに最後の選択肢として含まれる

**生成例**:
```cpp
Action pass;
pass.type = PlayerIntent::PASS;
```

---

### 2. MANA_CHARGE
**用途**: マナゾーンへのカード配置  
**生成フェーズ**: MANA  
**パラメータ**:
- `card_id`: チャージするカードID
- `source_instance_id`: カードインスタンスID
- `slot_index`: 手札内のインデックス

**説明**:
- 手札からマナゾーンにカードを1枚配置
- 1ターンに1回のみ（`mana_charged_by_player[]`フラグで管理）
- SimpleAI優先度: MANAフェーズで90、他フェーズで10

**生成条件**:
```cpp
if (!game_state.turn_stats.mana_charged_by_player[active_pid]) {
    for (card in hand) {
        MANA_CHARGE(card)
    }
}
```

---

### 3. MOVE_CARD (非推奨)
**用途**: 汎用カード移動  
**生成フェーズ**: なし（非推奨）  
**説明**: 
- レガシーアクション、現在は使用されていない
- 特定のアクション（MANA_CHARGE, PLAY_CARDなど）を使用すべき

---

### 4. PLAY_CARD
**用途**: カードプレイ（レガシー）  
**生成フェーズ**: なし（DECLARE_PLAYに置き換え）  
**説明**:
- レガシーアクション、現在はDECLARE_PLAYを使用
- 一部の古いコードで残存

---

### 5. PLAY_FROM_ZONE
**用途**: 任意ゾーンからのカードプレイ（統合ハンドラ）  
**生成フェーズ**: 特殊（効果からの発動）  
**説明**:
- 手札以外のゾーン（墓地、シールド、マナなど）からのプレイ
- 効果によるプレイで使用

---

### 6. ATTACK_PLAYER
**用途**: プレイヤー攻撃  
**生成フェーズ**: ATTACK  
**パラメータ**:
- `source_instance_id`: 攻撃クリーチャーのインスタンスID
- `target_player`: ターゲットプレイヤーID（opponent）
- `slot_index`: バトルゾーンでのインデックス
- `target_instance_id`: -1（プレイヤー攻撃マーク）

**説明**:
- クリーチャーによるプレイヤー直接攻撃
- SimpleAI優先度: ATTACKフェーズで85、他フェーズで10

**生成条件**:
```cpp
if (can_attack_player(creature) && !CANNOT_ATTACK) {
    ATTACK_PLAYER(creature, opponent)
}
```

---

### 7. ATTACK_CREATURE
**用途**: クリーチャー攻撃  
**生成フェーズ**: ATTACK  
**パラメータ**:
- `source_instance_id`: 攻撃クリーチャーのインスタンスID
- `target_instance_id`: ターゲットクリーチャーのインスタンスID
- `slot_index`: 攻撃側のバトルゾーンインデックス
- `target_slot_index`: 防御側のバトルゾーンインデックス

**説明**:
- タップされた相手クリーチャーへの攻撃
- ターゲットはタップ状態でなければならない
- Just Diver保護を受けているクリーチャーは攻撃不可

**生成条件**:
```cpp
if (can_attack_creature(creature) && !CANNOT_ATTACK) {
    for (tapped_enemy if !protected_by_just_diver) {
        ATTACK_CREATURE(creature, tapped_enemy)
    }
}
```

---

### 8. BLOCK
**用途**: ブロッカー宣言  
**生成フェーズ**: BLOCK  
**パラメータ**:
- `source_instance_id`: ブロッカークリーチャーのインスタンスID
- `slot_index`: バトルゾーンでのインデックス

**説明**:
- BLOCKERキーワードを持つクリーチャーによるブロック宣言
- アンタップ状態のクリーチャーのみ
- SimpleAI優先度: BLOCKフェーズで85、他フェーズで10

**生成条件**:
```cpp
for (blocker in defender.battle_zone if !tapped && has_BLOCKER) {
    if (!CANNOT_BLOCK) {
        BLOCK(blocker)
    }
}
```

---

### 9. USE_SHIELD_TRIGGER
**用途**: シールドトリガー使用  
**生成フェーズ**: pending_effects内（SHIELD_TRIGGER効果）  
**パラメータ**:
- `source_instance_id`: S・トリガーカードのインスタンスID
- `slot_index`: pending_effectsのインデックス

**説明**:
- シールドブレイク時に発動したS・トリガーカードを使用
- 任意使用（使わない選択も可能）

**生成例**:
```cpp
use.type = PlayerIntent::USE_SHIELD_TRIGGER;
use.source_instance_id = eff.source_instance_id;
```

---

### 10. SELECT_TARGET
**用途**: ターゲット選択  
**生成フェーズ**: クエリ応答（waiting_for_user_input）  
**パラメータ**:
- `target_instance_id`: 選択したターゲットのインスタンスID

**説明**:
- 効果実行時のターゲット選択
- クエリ応答として最優先（SimpleAI優先度: 100）
- valid_targetsリストから選択

**生成例**:
```cpp
if (query_type == "SELECT_TARGET") {
    for (target_id in valid_targets) {
        SELECT_TARGET(target_id)
    }
}
```

---

### 11. RESOLVE_EFFECT
**用途**: エフェクト解決  
**生成フェーズ**: pending_effects内  
**パラメータ**:
- `slot_index`: pending_effectsのインデックス
- `source_instance_id`: 効果ソースのインスタンスID

**説明**:
- pending_effects内の効果を解決
- 必須効果: SimpleAI優先度100
- オプショナル効果: SimpleAI優先度95（提案）
- フェーズアクションより常に優先

**生成例**:
```cpp
resolve.type = PlayerIntent::RESOLVE_EFFECT;
resolve.slot_index = effect_index;
resolve.source_instance_id = eff.source_instance_id;
```

---

### 12. USE_ABILITY
**用途**: 能力使用  
**生成フェーズ**: 特殊（Revolution Change, Ninja Strikeなど）  
**パラメータ**:
- `source_instance_id`: 能力を持つカードのインスタンスID

**説明**:
- Revolution Change、Ninja Strikeなどの特殊能力発動
- 条件を満たした場合に生成

---

### 13. DECLARE_REACTION
**用途**: リアクション宣言  
**生成フェーズ**: REACTION_WINDOW内  
**パラメータ**:
- `source_instance_id`: リアクションを行うカードのインスタンスID

**説明**:
- Ninja Strike、Strike Backなどのリアクション能力
- REACTION_WINDOWでのみ生成

**生成例**:
```cpp
act.type = PlayerIntent::DECLARE_REACTION;
act.source_instance_id = ninja_strike_card.instance_id;
```

---

### 14. SELECT_OPTION
**用途**: オプション選択  
**生成フェーズ**: クエリ応答（SELECT_OPTIONクエリ）、pending_effects内  
**パラメータ**:
- `target_slot_index`: 選択したオプションのインデックス

**説明**:
- モード選択、複数効果からの選択
- クエリ応答時: SimpleAI優先度100
- pending_effects内: 各効果に応じて生成

**生成例**:
```cpp
if (query_type == "SELECT_OPTION") {
    for (i in 0..options.size()) {
        SELECT_OPTION(option_index=i)
    }
}
```

---

### 15. SELECT_NUMBER
**用途**: 数値選択  
**生成フェーズ**: pending_effects内（SELECT_NUMBER効果）  
**パラメータ**:
- `target_slot_index`: 選択した数値

**説明**:
- 「X枚ドロー」などの数値選択
- 有効範囲内の数値を選択

**生成例**:
```cpp
select.type = PlayerIntent::SELECT_NUMBER;
select.target_slot_index = number_value;
```

---

## 🔧 エンジン内部アクション（Engine/Internal Actions）

### 16. DECLARE_PLAY
**用途**: カードプレイ宣言（Atomicフロー開始）  
**生成フェーズ**: MAIN  
**パラメータ**:
- `card_id`: プレイするカードID
- `source_instance_id`: カードインスタンスID
- `slot_index`: 手札内のインデックス
- `is_spell_side`: Twinpactスペル側か（bool）
- `target_slot_index`: Active Payment時の支払いユニット数

**説明**:
- カードプレイの開始アクション
- スタックゾーンにカードを配置
- PAY_COST → RESOLVE_PLAYの流れに続く
- SimpleAI優先度: MAINフェーズで80

**生成条件**:
```cpp
if (can_pay_cost && !spell_restricted) {
    DECLARE_PLAY(card, is_spell_side=false)
    
    // Twinpact Spell Side
    if (twinpact && can_pay_spell_side) {
        DECLARE_PLAY(card, is_spell_side=true)
    }
    
    // Active Cost Reduction (Hyper Energy)
    for (units in 1..max_units) {
        DECLARE_PLAY(card, payment_units=units)
    }
}
```

---

### 17. PAY_COST
**用途**: マナコスト支払い（Atomic）  
**生成フェーズ**: Stack処理  
**パラメータ**:
- `source_instance_id`: スタック上のカードインスタンスID
- `card_id`: カードID

**説明**:
- スタックゾーンのカードのコスト支払い
- マナゾーンからカードをタップ
- SimpleAI優先度: 98

**生成条件**:
```cpp
for (card in stack if !is_tapped && can_pay_cost) {
    PAY_COST(card)
}
```

---

### 18. RESOLVE_PLAY
**用途**: プレイ解決（Atomic）  
**生成フェーズ**: Stack処理、pending_effects内  
**パラメータ**:
- `source_instance_id`: スタック上のカードインスタンスID
- `card_id`: カードID

**説明**:
- コスト支払い済みカードを実際にプレイ
- Creature → バトルゾーンへ
- Spell → 効果発動後墓地へ
- SimpleAI優先度: 98

**生成条件**:
```cpp
// Stack処理
for (card in stack if is_tapped) {
    RESOLVE_PLAY(card)
}

// pending_effects内のINTERNAL_PLAY
if (pending.type == INTERNAL_PLAY) {
    RESOLVE_PLAY(pending.source_instance_id)
}
```

---

### 19. PLAY_CARD_INTERNAL
**用途**: スタック内部プレイアクション  
**生成フェーズ**: pending_effects内（INTERNAL_PLAY効果）  
**パラメータ**:
- `source_instance_id`: プレイするカードのインスタンスID
- `spawn_source`: SpawnSource（HAND_SUMMON, EFFECT_SUMMON, EFFECT_PUT）

**説明**:
- Gatekeeperなどの効果によるカードプレイ
- pending_effects内で生成
- SimpleAI優先度: 80（PLAY_CARDと同等）

**生成例**:
```cpp
action.type = PlayerIntent::PLAY_CARD_INTERNAL;
action.source_instance_id = eff.source_instance_id;
action.spawn_source = SpawnSource::EFFECT_SUMMON;
```

---

### 20. RESOLVE_BATTLE
**用途**: バトル解決  
**生成フェーズ**: pending_effects内（RESOLVE_BATTLE効果）  
**パラメータ**:
- `slot_index`: pending_effectsのインデックス

**説明**:
- 攻撃・ブロック後のパワー比較と破壊判定
- pending_effectsから自動生成
- SimpleAI優先度: 100（必須効果）

**生成例**:
```cpp
action.type = PlayerIntent::RESOLVE_BATTLE;
action.slot_index = effect_index;
```

---

### 21. BREAK_SHIELD
**用途**: シールドブレイク  
**生成フェーズ**: pending_effects内（BREAK_SHIELD効果）  
**パラメータ**:
- `slot_index`: pending_effectsのインデックス

**説明**:
- プレイヤー攻撃成功時のシールドブレイク処理
- pending_effectsから自動生成
- SimpleAI優先度: 100（必須効果）

**生成例**:
```cpp
action.type = PlayerIntent::BREAK_SHIELD;
action.slot_index = effect_index;
```

---

## 📊 アクション使用頻度分類

### 🔴 最頻出（毎ゲーム複数回）
```
PASS, MANA_CHARGE, DECLARE_PLAY, PAY_COST, RESOLVE_PLAY, RESOLVE_EFFECT
```

### 🟡 頻出（ゲーム中数回）
```
ATTACK_PLAYER, SELECT_TARGET, ATTACK_CREATURE, BLOCK
```

### 🟢 まれ（条件付き）
```
USE_SHIELD_TRIGGER, SELECT_OPTION, DECLARE_REACTION, SELECT_NUMBER,
RESOLVE_BATTLE, BREAK_SHIELD, PLAY_CARD_INTERNAL
```

### ⚪ 非推奨/未使用
```
MOVE_CARD, PLAY_CARD, PLAY_FROM_ZONE, USE_ABILITY
```

---

## 🔍 SimpleAI優先度対応表

### 現在の実装（Phase 1）
[src/engine/ai/simple_ai.cpp](src/engine/ai/simple_ai.cpp)

| アクション | 優先度 | 問題点 |
|-----------|--------|--------|
| RESOLVE_EFFECT | 100 | ✅ 正しい |
| SELECT_TARGET | 90 | ✅ 正しい |
| SELECT_OPTION | 90 | ✅ 正しい |
| PLAY_CARD | 80 | ⚠️ フェーズ非対応 |
| DECLARE_BLOCKER | 70 | ❓ 未定義（BLOCKと混同？） |
| DECLARE_NO_BLOCK | 20 | ❓ 未定義 |
| ATTACK | 60 | ⚠️ フェーズ非対応 |
| MANA_CHARGE | 40 | ❌ MANAフェーズで低すぎ |
| PASS | 0 | ✅ 正しい |

**注**: DECLARE_BLOCKER、DECLARE_NO_BLOCKはPlayerIntent定義に存在せず、BLOCKアクションの誤記と思われる

---

## 🎯 推奨優先度（Phase 1.1改善版）

### Level 1: 必須応答（100）
```
SELECT_TARGET, SELECT_OPTION, RESOLVE_EFFECT (必須), 
RESOLVE_BATTLE, BREAK_SHIELD
```

### Level 2: 高優先度（95-98）
```
PAY_COST (98), RESOLVE_PLAY (98), RESOLVE_EFFECT (オプショナル, 95)
```

### Level 3: フェーズアクション（80-90）
```
MANA_CHARGE (MANAフェーズ: 90)
ATTACK_PLAYER/CREATURE (ATTACKフェーズ: 85)
BLOCK (BLOCKフェーズ: 85)
DECLARE_PLAY (MAINフェーズ: 80)
PLAY_CARD_INTERNAL (80)
```

### Level 4: 低優先度（10-50）
```
USE_SHIELD_TRIGGER (50: オプショナル)
DECLARE_REACTION (50)
Other Actions (20)
Wrong Phase Actions (10)
```

### Level 5: 最低（0）
```
PASS (0)
```

---

## 📋 Action構造体パラメータ

```cpp
struct Action {
    PlayerIntent type;              // アクション種別
    CardID card_id;                 // カードID (PLAY, MANA_CHARGE用)
    int source_instance_id;         // ソースカードのインスタンスID
    int target_instance_id;         // ターゲットのインスタンスID
    PlayerID target_player;         // ターゲットプレイヤー (ATTACK_PLAYER用)
    
    int slot_index;                 // ソースのスロットインデックス
    int target_slot_index;          // ターゲットのスロットインデックス
    
    SpawnSource spawn_source;       // PLAY_CARD_INTERNAL用のSpawnSource
    bool is_spell_side;             // Twinpactスペル側フラグ
};
```

---

## 🔗 関連ドキュメント

- [PHASE_ACTION_PRIORITY_SPEC.md](PHASE_ACTION_PRIORITY_SPEC.md) - フェーズ別優先度仕様
- [PRIORITY_QUICK_REFERENCE.md](PRIORITY_QUICK_REFERENCE.md) - 優先度クイックリファレンス
- [PHASE_AWARE_AI_DESIGN.md](PHASE_AWARE_AI_DESIGN.md) - フェーズ対応AI設計案

---

**ドキュメント作成日**: 2026年2月7日  
**バージョン**: 1.0  
**定義ファイル**: [src/core/action.hpp](src/core/action.hpp)  
**総アクション数**: 22種類（ユーザー15 + エンジン7）
