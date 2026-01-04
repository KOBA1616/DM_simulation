# Game Engine Specifications (要件定義書 01)

## 1. コアアーキテクチャ (Core Architecture)

### 1.1 Game State & Command System
*   **State Management**: `GameState` クラスを中心としたデータ駆動型設計。
*   **Command Pattern**: すべての状態遷移は `GameCommand` (Transition, Mutate, Flow) を介して行われる。
    *   `MutateCommand`: カード移動、パラメータ変更、Modifier適用など。
    *   `FlowCommand`: フェーズ遷移、ターン終了処理など。
*   **Hash & Loop Detection**: `GameState::calculate_hash` による盤面ハッシュ計算と、それを用いたループ検出 (`GameStateLoop`)。

### 1.2 Action System
*   **PendingEffect Queue**: `TRIGGER_ABILITY`, `RESOLVE_BATTLE` 等の処理待ちイベントを管理するスタック。
*   **Strategies**: `ActionGenerator` は `MainPhaseStrategy`, `AttackPhaseStrategy` 等のストラテジーに処理を委譲。
*   **Effect Resolver**: `EffectResolver` および `IActionHandler` によるアクション解決のモジュール化。

## 2. 実装済みメカニクス (Implemented Mechanics)

### 2.1 カード属性・キーワード
*   **Multi-Civilization**: `std::vector<Civilization>` による多色対応。マナ解放条件（少なくとも1枚のマナが必要）を `ManaSystem` で処理。
*   **Hyper Energy (ハイパー化)**: クリーチャーをタップしてコスト軽減する `ActionType::PLAY_CARD` の特殊バリアント。
*   **Revolution Change (革命チェンジ)**: `EffectResolver` および `TriggerType::ON_ATTACK_FROM_HAND` による攻撃時の入れ替わり処理。
*   **Just Diver (ジャストダイバー)**: `CardInstance.turn_played` と `GameState.turn_number` の比較によるターゲット保護。

### 2.2 ゾーン・処理
*   **Zone Management**: Hand, Deck, Mana, Battle, Graveyard, Shield, Stack, EffectBuffer (一時領域)。
*   **Variable Linking**: 前のアクションの結果（`GET_GAME_STAT` 等）を次のアクション（`DRAW_CARD` 等）の入力として連鎖させる `execution_context` システム。

## 3. 今後の改修要件 (Refactoring Requirements)

### 3.1 技術的負債の解消
*   **Legacy Code**: `src/engine/game_logic` 内に残存する古いロジックを `IActionHandler` へ完全移行する。
*   **Test Coverage**: `tests/` 内のC++単体テストを拡充し、特に複雑な処理（革命チェンジの条件判定など）をカバーする。

## 4. API & Bindings
*   **Python Integration**: `pybind11` を用いて `dm_ai_module` としてビルド。
*   **Direct Access**: `GameState`, `CardDefinition`, `ActionGenerator` 等のコアクラスへの直接アクセスを提供し、Python側での柔軟な学習・テストを可能にする。
