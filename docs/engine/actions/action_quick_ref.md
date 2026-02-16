# PlayerIntentアクション クイックリファレンス

## 📋 全22種類 一覧表

| # | アクション名 | カテゴリ | 生成フェーズ | 優先度 | 頻度 |
|---|-------------|---------|-------------|--------|------|
| 1 | PASS | ユーザー | すべて | 0 | 🔴 |
| 2 | MANA_CHARGE | ユーザー | MANA | 90 | 🔴 |
| 3 | ~~MOVE_CARD~~ | ~~非推奨~~ | - | - | ⚪ |
| 4 | ~~PLAY_CARD~~ | ~~レガシー~~ | - | - | ⚪ |
| 5 | PLAY_FROM_ZONE | ユーザー | 特殊 | - | 🟢 |
| 6 | ATTACK_PLAYER | ユーザー | ATTACK | 85 | 🟡 |
| 7 | ATTACK_CREATURE | ユーザー | ATTACK | 85 | 🟡 |
| 8 | BLOCK | ユーザー | BLOCK | 85 | 🟡 |
| 9 | USE_SHIELD_TRIGGER | ユーザー | pending | 50 | 🟢 |
| 10 | SELECT_TARGET | ユーザー | query | 100 | 🟡 |
| 11 | RESOLVE_EFFECT | ユーザー | pending | 100/95 | 🔴 |
| 12 | USE_ABILITY | ユーザー | 特殊 | 50 | 🟢 |
| 13 | DECLARE_REACTION | ユーザー | reaction | 50 | 🟢 |
| 14 | SELECT_OPTION | ユーザー | query/pending | 100 | 🟡 |
| 15 | SELECT_NUMBER | ユーザー | pending | 100 | 🟢 |
| 16 | **DECLARE_PLAY** | **内部** | **MAIN** | **80** | **🔴** |
| 17 | **PAY_COST** | **内部** | **Stack** | **98** | **🔴** |
| 18 | **RESOLVE_PLAY** | **内部** | **Stack** | **98** | **🔴** |
| 19 | **PLAY_CARD_INTERNAL** | **内部** | **pending** | **80** | **🟢** |
| 20 | **RESOLVE_BATTLE** | **内部** | **pending** | **100** | **🟢** |
| 21 | **BREAK_SHIELD** | **内部** | **pending** | **100** | **🟢** |

**頻度**: 🔴 最頻出 | 🟡 頻出 | 🟢 まれ | ⚪ 非推奨

---

## 🎯 優先度別分類

### Level 1: 必須応答（100）
```
SELECT_TARGET, SELECT_OPTION, SELECT_NUMBER
RESOLVE_EFFECT (必須), RESOLVE_BATTLE, BREAK_SHIELD
```

### Level 2: 高優先度（95-98）
```
PAY_COST (98), RESOLVE_PLAY (98)
RESOLVE_EFFECT (オプション, 95)
```

### Level 3: フェーズアクション（80-90）
```
MANA_CHARGE (90)
ATTACK_PLAYER, ATTACK_CREATURE, BLOCK (85)
DECLARE_PLAY, PLAY_CARD_INTERNAL (80)
```

### Level 4: 低優先度（10-50）
```
USE_SHIELD_TRIGGER, USE_ABILITY, DECLARE_REACTION (50)
Other (20), Wrong Phase (10)
```

### Level 5: 最低（0）
```
PASS (0)
```

---

## 📊 フェーズ別生成アクション

### MANA Phase
```
✅ MANA_CHARGE (90)
✅ PASS (0)
```

### MAIN Phase
```
✅ DECLARE_PLAY (80)
   ├─ 通常プレイ
   ├─ Twinpactスペル側
   └─ Active Cost Reduction
✅ PASS (0)
```

### ATTACK Phase
```
✅ ATTACK_PLAYER (85)
✅ ATTACK_CREATURE (85)
✅ PASS (0)
```

### BLOCK Phase
```
✅ BLOCK (85)
✅ PASS (0)
```

### Stack Processing
```
✅ PAY_COST (98)
✅ RESOLVE_PLAY (98)
```

### Pending Effects
```
✅ RESOLVE_EFFECT (100/95)
✅ SELECT_TARGET (100)
✅ USE_SHIELD_TRIGGER (50)
✅ RESOLVE_BATTLE (100)
✅ BREAK_SHIELD (100)
✅ PLAY_CARD_INTERNAL (80)
✅ DECLARE_REACTION (50)
```

### Query Response
```
✅ SELECT_TARGET (100)
✅ SELECT_OPTION (100)
✅ SELECT_NUMBER (100)
```

---

## 🔄 Atomicフロー

```
DECLARE_PLAY → PAY_COST → RESOLVE_PLAY → RESOLVE_EFFECT
     (80)        (98)         (98)          (100/95)
```

---

## 🔗 関連リンク

- [完全リファレンス](../PLAYER_INTENT_REFERENCE.md) - 全アクション詳細
- [アクション分類図](action_classification.md) - Mermaid図
- [生成フロー図](action_generation_flow.md) - フロー図
- [優先度仕様](../PHASE_ACTION_PRIORITY_SPEC.md) - フェーズ別優先度

---

**最終更新**: 2026年2月7日
