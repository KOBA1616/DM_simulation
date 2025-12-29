# 9. 汎用カードジェネレーターとデータ駆動アーキテクチャ (Generic Card Generator & Data-Driven Architecture)

## 9.1 概要 (Overview)
本セクションは、カードの実装を「プログラム（コーディング）」から「データ定義（コンフィグレーション）」へと移行するための設計指針である。Unity等のゲームエンジンで採用される **Entity-Component-System (ECS)** パターン及び **バイトコードインタプリタ** の概念をTCG向けに軽量化し、90%以上のカードを再コンパイルなしで実装可能にすることを目的とする。

- **Architecture**:
    1.  **Definition (データ層)**: カードの挙動を定義したJSONデータ。
    2.  **Generator (ツール層)**: GUI/LLMを用いてDefinitionを生成する外部ツール。
    3.  **Interpreter (エンジン層)**: Definitionを読み込み、動的にルール処理を行うC++クラス。

## 9.2 汎用データ構造設計 (Data Structures)
カード効果を最小単位（Atom）に分解し、それらを組み合わせることで複雑な効果を表現する。

### 9.2.1 基本語彙 (Vocabulary Enums)
- **TriggerType**: `ON_PLAY` (CIP), `ON_ATTACK`, `ON_DESTROY` (PIG), `S_TRIGGER`, `TURN_START`, `PASSIVE_CONST` 等。
- **TargetScope**: `SELF`, `PLAYER_SELF`, `PLAYER_OPPONENT`, `ALL_PLAYERS`, `TARGET_SELECT`, `RANDOM`, `ALL_FILTERED`.
- **ActionType**: `DRAW_CARD`, `ADD_MANA`, `DESTROY`, `RETURN_TO_HAND`, `SEND_TO_MANA`, `TAP`, `UNTAP`, `MODIFY_POWER`, `BREAK_SHIELD`, `LOOK_AND_ADD`, `SUMMON_TOKEN`.

### 9.2.2 構造体定義
- **FilterDef**: 対象を選択するための条件（Type, Civ, Race, Cost/Power Range, Tapped/Blocker State）。
- **ActionDef**: 1つの具体的行動。`ActionType`, パラメータ(`value1`, `value2`, `str_val`), 対象(`TargetScope`, `FilterDef`)を持つ。
- **ConditionDef**: 発動条件（例: マナ武装5）。
- **EffectDef**: 1つの効果ブロック。`TriggerType`, `ConditionDef`, `std::vector<ActionDef>` (順次実行) を持つ。
- **CardData**: カード本体の定義。ID, Name, Cost, Power, Civ, Races, Type, `std::vector<EffectDef>`。

## 9.3 JSONスキーマ設計 (Serialization)
C++の構造体と1:1で対応するJSONフォーマット。
例: 「破壊とドローの悪魔」 (登場時、相手のアンタップ獣1体を破壊し、1ドロー)

```json
{
  "id": 205,
  "name": "Demon of Destruction and Draw",
  "cost": 5,
  "civilization": "DARKNESS",
  "power": 5000,
  "type": "CREATURE",
  "races": ["Demon Command"],
  "effects": [
    {
      "trigger": "ON_PLAY",
      "condition": { "type": "NONE" },
      "actions": [
        {
          "type": "DESTROY",
          "scope": "TARGET_SELECT",
          "filter": { "owner": "OPPONENT", "types": ["CREATURE"], "is_tapped": false }
        },
        {
          "type": "DRAW_CARD",
          "scope": "PLAYER_SELF",
          "value1": 1
        }
      ]
    }
  ]
}
```

## 9.4 ジェネレーターツール設計 (Tool Design)
開発者がGUI操作でカードを登録するためのツール。
- **Basic Info Area**: コスト、パワー、文明などの入力フォーム。
- **Effect Builder Area**: ノードベースまたはリストベースのエディタ。
- **AI Assistant**: 自然言語を入力するとJSONを自動生成するプロンプト入力欄。
- **JSON Preview**: リアルタイムで生成されるJSONを表示。

## 9.5 C++エンジン実装 (Interpreter Implementation)
個別のカードクラスを作らず、`GenericCard` クラスがデータ（`CardData`）を解釈して動作する。
また、**イベント駆動型アーキテクチャ (Event-Driven Architecture)** を採用し、拡張性を高める。

```cpp
class GenericCard : public Card {
    const CardData* data; // マスタデータ参照
public:
    GenericCard(const CardData* d) : data(d) { /* 初期化 */ }

    // イベントリスナー形式での実装
    void on_event(EventType type, const EventContext& ctx) override {
        for (const auto& effect : data->effects) {
            // トリガー条件の詳細チェック (例: 破壊されたのが自分か？ マナに置かれたカードは火文明か？)
            if (!effect.matches_trigger(type, ctx)) continue;

            if (!check_condition(effect.condition, ctx.game_state)) continue;

            for (const auto& action : effect.actions) {
                resolve_action(action, ctx.game_state);
            }
        }
    }
private:
    void resolve_action(const ActionDef& action, GameState& state) {
        auto targets = Selector::select(action.scope, action.filter, state, this);
        switch (action.type) {
            case ActionType::DESTROY: for (auto* t : targets) state.destroy(t); break;
            case ActionType::DRAW_CARD: state.draw(this->owner_id, action.value1); break;
            // ... 他のアクション
        }
    }
};
```
- **Supported Events**: `OnPreDestroy`, `OnManaZonePut`, `OnShieldBreakReplacement` 等、細かいフックポイントを用意することで、置換効果や常在型能力にも対応する。

## 9.6 拡張性と制限 (Extensibility)
- **Coverage**: バニラカード、単純な除去、ドロー、マナ加速、パワー修正など、カードプールの約80〜90%をカバー可能。
- **Hybrid Approach**: 「エクストラウィン」や「無限ループを伴う複雑な置換効果」などは、従来のC++クラス継承 (`class SpecialWinCard : public Card`) を併用する。ジェネレーターツールで「Custom C++ Implementation」を選択可能にする。
