# 呪文のゾーン経路と置換効果の実装ガイド

## 1. 通常の呪文解決フロー

### 1.1 ゾーン経路（置換効果なし）

```
手札（HAND）
    ↓ [プレイ宣言: DECLARE_PLAY]
スタック（STACK）
    ↓ [コスト支払い: PAY_COST]
スタック（STACK - tapped）
    ↓ [解決: RESOLVE_PLAY]
効果実行
    ↓ [自動移動]
墓地（GRAVEYARD）
```

### 1.2 実装の詳細

**エンジンコード**: `src/engine/systems/game_logic_system.cpp`

```cpp
void GameLogicSystem::handle_resolve_play(...) {
    // ... 効果コンパイル ...
    
    if (def.type == CardType::SPELL) {
        // 1. 呪文効果をコンパイル
        for (const auto& eff : def.effects) {
            EffectSystem::instance().compile_effect(
                state, eff, instance_id, ctx, card_db, compiled_effects
            );
        }
        
        // 2. スタックから墓地へ移動（効果の後）
        nlohmann::json move_args;
        move_args["target"] = instance_id;
        move_args["to"] = "GRAVEYARD";
        compiled_effects.emplace_back(InstructionOp::MOVE, move_args);
    }
}
```

**重要ポイント**:
- 呪文の効果はすべてコンパイルされた後、最後に自動的に `GRAVEYARD` への移動命令が追加される
- この移動は `STACK` から `GRAVEYARD` への `TransitionCommand` として実行される

---

## 2. 置換効果を使用した場合

### 2.1 ゾーン経路（置換効果あり）

```
手札（HAND）
    ↓ [プレイ宣言]
スタック（STACK）
    ↓ [コスト支払い]
スタック（STACK - tapped）
    ↓ [解決]
効果実行
    ↓ [置換効果が介入]
墓地（GRAVEYARD）に行く代わりに → 山札の下（DECK_BOTTOM）
```

### 2.2 実装方法

#### 方法1: エフェクトレベルでの置換（推奨）

カード定義JSONで、呪文効果の最後に `REPLACE_CARD_MOVE` を追加します：

```json
{
  "name": "置換呪文サンプル",
  "type": "SPELL",
  "civilization": "WATER",
  "cost": 3,
  "effects": [
    {
      "type": "DRAW_CARD",
      "amount": 2,
      "target_group": "PLAYER_SELF"
    },
    {
      "type": "REPLACE_CARD_MOVE",
      "from_zone": "GRAVEYARD",
      "to_zone": "DECK_BOTTOM",
      "target_group": "SELF",
      "comment": "この呪文を墓地に置く代わりに山札の下に置く"
    }
  ]
}
```

**処理フロー**:
1. 呪文効果（DRAW_CARD）が実行される
2. エンジンが自動的に `STACK → GRAVEYARD` への移動を追加
3. `REPLACE_CARD_MOVE` 効果が実行され、墓地への移動を検出
4. 代わりに `STACK → DECK_BOTTOM` への移動を実行

#### 方法2: トリガーベースの置換

呪文解決後のトリガーで置換を実装：

```json
{
  "name": "トリガー置換呪文",
  "type": "SPELL",
  "effects": [
    {
      "type": "DRAW_CARD",
      "amount": 1
    }
  ],
  "triggers": [
    {
      "type": "ON_ZONE_ENTER",
      "zone": "GRAVEYARD",
      "condition": {
        "type": "SELF_TARGET"
      },
      "effects": [
        {
          "type": "TRANSITION",
          "from_zone": "GRAVEYARD",
          "to_zone": "DECK_BOTTOM",
          "target_group": "SELF"
        }
      ]
    }
  ]
}
```

---

## 3. エンジンでの置換効果実装

### 3.1 TransitionCommand の実行

**コード**: `src/engine/game_command/commands.cpp`

```cpp
void TransitionCommand::execute(GameState& state) {
    // 1. カードを元のゾーンから取り出す
    CardInstance card = remove_from_zone(from_zone);
    
    // 2. 置換効果のチェック（今後の拡張ポイント）
    // if (replacement_effects_exist()) {
    //     Zone new_destination = check_replacement_effects(card, to_zone);
    //     to_zone = new_destination;
    // }
    
    // 3. 新しいゾーンに配置
    add_to_zone(to_zone, card);
    
    // 4. イベント発火
    dispatch_event(ZONE_ENTER, to_zone, card);
}
```

### 3.2 置換効果の検証

**compat.py** での処理（Python側フォールバック）:

```python
if ctype == 'REPLACE_CARD_MOVE':
    # 本来の移動先
    original_zone = cd.get('original_to_zone') or cd.get('from_zone')
    # 実際の移動先
    dest_zone = cd.get('to_zone') or 'DECK_BOTTOM'
    instance_id = cd.get('instance_id')
    
    # カードを取得
    card_obj, pid = _detach_instance(instance_id)
    
    # 新しいゾーンに配置（墓地をスキップ）
    _place(card_obj, pid, dest_zone)
```

---

## 4. 重要な保証事項

### 4.1 墓地に置かれないことの保証

`REPLACE_CARD_MOVE` を使用する場合：

1. **エンジンレベル**: `TransitionCommand` が置換先ゾーンに直接移動
   - 墓地ゾーンのリストには追加されない
   - `ZONE_ENTER` イベントは置換先ゾーンでのみ発火

2. **トリガーの回避**: 
   - "墓地に置かれた時" のトリガーは発動しない
   - 置換先ゾーンの "置かれた時" トリガーが発動

3. **検証方法**:
```python
# テストコード例
def test_spell_replacement_skips_graveyard():
    # 呪文を唱える
    play_spell(spell_card_id)
    
    # 墓地を確認（空であるべき）
    assert len(state.players[0].graveyard) == 0
    
    # 山札の下を確認
    assert state.players[0].deck[0].card_id == spell_card_id
```

---

## 5. 実装例：完全なカード定義

### 5.1 墓地に置かれない呪文

```json
{
  "card_id": 9001,
  "name": "時空の霧",
  "type": "SPELL",
  "civilization": "WATER",
  "cost": 4,
  "text": "カードを2枚引く。この呪文を唱えた後、墓地に置く代わりに山札の下に置く。",
  "effects": [
    {
      "type": "DRAW_CARD",
      "amount": 2,
      "target_group": "PLAYER_SELF"
    }
  ],
  "on_resolve_replacement": {
    "type": "REPLACE_CARD_MOVE",
    "from_zone": "GRAVEYARD",
    "to_zone": "DECK_BOTTOM",
    "target_group": "SELF"
  }
}
```

### 5.2 GUIエディタでの設定

カードエディタで以下のように設定：

1. **基本情報**
   - タイプ: SPELL
   - 文明: 水
   - コスト: 4

2. **効果1: カードを引く**
   - アクショングループ: `DRAW`
   - コマンドタイプ: `DRAW_CARD`
   - Amount: 2
   - Target Group: PLAYER_SELF

3. **効果2: 置換移動**
   - アクショングループ: `CARD_MOVE`
   - コマンドタイプ: `REPLACE_CARD_MOVE`
   - From Zone: `GRAVEYARD`（本来行くゾーン）
   - To Zone: `DECK_BOTTOM`（実際の移動先）
   - Target Group: `SELF`

**生成テキスト**:
```
カードを2枚引く。
このカードを墓地に置くかわりに山札の下に置く。
```

---

## 6. デバッグとトラブルシューティング

### 6.1 ゾーン追跡

EngineCompat はデバッグ出力を提供します：

```
EngineCompat: REPLACE_CARD_MOVE moved 1 card(s) from GRAVEYARD to DECK_BOTTOM
```

### 6.2 一般的な問題

#### 問題1: 呪文が墓地に残る

**原因**: `REPLACE_CARD_MOVE` が実行される前に墓地移動が完了している

**解決策**: 
- `on_resolve_replacement` フィールドを使用
- または、トリガーの優先度を確認

#### 問題2: 効果が実行されない

**原因**: 置換効果が呪文効果を上書きしている

**解決策**:
- 呪文効果を先にコンパイル
- 置換効果は最後に実行

### 6.3 テスト戦略

```python
def test_spell_with_replacement():
    """呪文が墓地に行かず、山札の下に置かれることを確認"""
    
    # 初期状態
    initial_deck_size = len(state.players[0].deck)
    
    # 呪文を唱える
    cast_spell(spell_instance_id)
    
    # 検証
    assert len(state.players[0].graveyard) == 0, "墓地は空であるべき"
    assert len(state.players[0].deck) == initial_deck_size + 1, "山札に1枚追加"
    assert state.players[0].deck[0].instance_id == spell_instance_id, "山札の下に配置"
```

---

## 7. まとめ

### 通常フロー
```
HAND → STACK → [効果] → GRAVEYARD
```

### 置換フロー
```
HAND → STACK → [効果] → [REPLACE_CARD_MOVE] → DECK_BOTTOM
                                  ↓
                            (GRAVEYARDをスキップ)
```

### キーポイント

1. ✅ **墓地に置かれない**: `REPLACE_CARD_MOVE` は直接置換先に移動
2. ✅ **トリガー回避**: 墓地関連のトリガーは発動しない
3. ✅ **効果順序**: 呪文効果 → 置換効果の順で実行
4. ✅ **検証可能**: テストで墓地が空であることを確認可能

### 関連ファイル

- エンジン: `src/engine/systems/game_logic_system.cpp` (handle_resolve_play)
- コマンド: `src/engine/game_command/commands.cpp` (TransitionCommand)
- Python互換: `dm_toolkit/engine/compat.py` (REPLACE_CARD_MOVE処理)
- アクション変換: `dm_toolkit/action_to_command.py` (_handle_replace_card_move)
- UI設定: `dm_toolkit/gui/editor/configs/action_config.py`
