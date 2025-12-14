# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。
現在、Phase 0（基盤構築）、Phase 1（エディタ・エンジン拡張）および Phase 2（不完全情報対応）の主要機能、Phase 3（自己学習エコシステム）のコア実装を完了し、**Phase 4（アーキテクチャ刷新）およびシステム全体の安定化・リファクタリング**へフェーズを移行しています。

Python側のコードベースは `dm_toolkit` パッケージとして再構築され、モジュール性が向上しました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **フルスペック実装**: 基本ルールに加え、革命チェンジ、侵略、ハイパーエナジー、ジャストダイバー、ツインパクト、封印（基礎）、呪文ロックなどの高度なメカニクスをサポート済み。
*   **整合性と安定性の向上 (New)**:
    *   **データ構造の統一**: `CardDefinition` バインディングにおける非推奨プロパティ（`civilization`）を削除し、C++コアと同様に `civilizations`（リスト形式）の使用を強制することで、多色カードの扱いに関する潜在的なバグを排除しました。
    *   **型安全性の向上**: `EffectResolver` や `bindings.cpp` 内での不要な型チェック（符号なし整数に対する負値チェックなど）を削除し、コードの健全性を向上させました。
    *   **終了処理の安定化**: Python終了時のセグメンテーション違反（`gilstate_tss_set` エラー）の原因となっていたバッチ推論コールバックの解放漏れを修正し、明示的なクリーンアップ API (`clear_batch_inference`) を導入・適用しました。
*   **汎用コストシステム（統合完了）**: `CostPaymentSystem` を実装し、能動的コスト軽減（ハイパーエナジー等）の計算基盤およびアクション生成・実行ロジックをエンジンに統合しました。
*   **アクションシステム**: `IActionHandler` による完全なモジュラー構造。
    *   **カード移動の統一**: `MOVE_CARD` アクションを導入し、DRAW, ADD_MANA, DESTROYなどの専用アクションを汎用移動ロジックで表現可能にしました。
    *   **条件判定の汎用化**: `COMPARE_STAT` 条件を導入し、「手札枚数がX以上」などの数値比較条件を自由に記述可能にしました。
    *   **変数リンクの完全サポート**: 各アクションハンドラ（Draw, Destroy, Shield等）において、`input_value_key` による入力値の参照と `output_value_key` による実行結果の書き込み（例：破壊した数の出力）をサポートしました。
    *   **モード選択機能**: `SELECT_OPTION` アクションを導入し、「選択肢から選ぶ」効果の実装と、ネストされたアクションのGUI編集に対応しました。
*   **高速シミュレーション**: OpenMPによる並列化と最適化されたメモリ管理により、秒間数千〜数万試合の自己対戦が可能。
*   **不完全情報探索 (PIMC)**: 推定された世界でのモンテカルロ探索（Determinization）を並列実行可能。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.3**: 3ペイン構成（ツリー/プロパティ/プレビュー）。
    *   **UI改善**: カードプレビューの視認性向上（黒枠化、マナコスト色分け、ツインパクトレイアウト修正）を実施済み。
    *   **テキスト生成の拡充 (New)**: `CardTextGenerator` を更新し、数値範囲（「1〜5の数字を選ぶ」）や任意選択（「〜まで選ぶ」「そうしてもよい」）の日本語生成ロジックを強化しました。また、ステータス参照条件の日本語化も拡充しました。
    *   **安定性**: プロパティインスペクタのクラッシュバグを修正済み。
    *   **不整合修正**: 革命チェンジのデータ構造、文明指定キーの不整合を解消済み。
    *   **リアクション編集**: `ReactionWidget` を最適化し、能力タイプに応じた動的なフィールド表示（コスト、ゾーン、トリガー条件の可視化）に対応しました。
*   **機能**: JSONデータの視覚的編集、ロジックツリー、変数リンク、テキスト自動生成、デッキビルダー、シナリオエディタ。
*   **検証済み**: 生成されたJSONデータはエンジンで即座に読み込み可能。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **AlphaZero Pipeline**: データ収集 -> 学習 -> 評価 の完全自動ループが稼働中。
*   **推論エンジン**: 相手デッキタイプ推定 (`DeckClassifier`) と手札確率推定 (`HandEstimator`) を実装済み。
*   **探索アルゴリズム**:
    *   **MCTS (Monte Carlo Tree Search)**: 標準的なUCTベースの探索。
    *   **Beam Search (ビームサーチ)**: `BeamSearchEvaluator` クラスを実装済み。幅(Beam Width)と深さ(Depth)を指定して、トリガーリスクや相手の脅威度（Opponent Danger）をヒューリスティックに評価しながら決定的な探索を行うことが可能です。
*   **自己進化**: 遺伝的アルゴリズムによるデッキ改良ロジック (`DeckEvolution`) がC++コアに統合され、高速に動作します。
*   **ONNX Runtime (C++) 統合**: 学習済みPyTorchモデルをONNX形式でエクスポートし、C++エンジン内で直接高速推論を行う `NeuralEvaluator` の拡張を完了しました。
*   **Phase 4 アーキテクチャ**: `NetworkV2` (Transformer/Linear Attention) の実装と、C++側のシーケンス変換ロジック (`TensorConverter::convert_to_sequence`) の実装・結合を完了しました。

### 2.4 サポート済みアクション・トリガー一覧 (Supported Actions & Triggers)

現在のコードベースで実装・登録されている主要な列挙子は以下の通りです。

**実装済みアクション (EffectActionType)**
以下のタイプは `IActionHandler` が登録されており、動作します。
*   **基本操作**: `DRAW_CARD`, `ADD_MANA` (Charge), `DESTROY`, `RETURN_TO_HAND`, `TAP`, `UNTAP`, `MOVE_CARD` (汎用移動), `CAST_SPELL`, `PUT_CREATURE`
*   **シールド操作**: `ADD_SHIELD`, `SEND_SHIELD_TO_GRAVE`, `BREAK_SHIELD` (Engine Loop / Action)
*   **山札操作**: `SEARCH_DECK`, `SEARCH_DECK_BOTTOM`, `SEND_TO_DECK_BOTTOM`, `SHUFFLE_DECK`, `REVEAL_CARDS`, `LOOK_AND_ADD` (Partial/Alias)
*   **バッファ・特殊領域**: `MEKRAID`, `LOOK_TO_BUFFER`, `MOVE_BUFFER_TO_ZONE`, `PLAY_FROM_BUFFER`
*   **数値・変数**: `COUNT_CARDS`, `GET_GAME_STAT` (マナ文明数、手札枚数等), `SELECT_NUMBER`, `COST_REFERENCE`, `APPLY_MODIFIER` (Power/Cost修正)
*   **その他**: `SELECT_OPTION` (モード選択), `FRIEND_BURST`, `GRANT_KEYWORD`, `MOVE_TO_UNDER_CARD`

**実装済みトリガー (TriggerType)**
*   **標準**: `ON_PLAY` (cip), `ON_ATTACK`, `ON_DESTROY`, `S_TRIGGER`, `TURN_START`, `PASSIVE_CONST`
*   **拡張**: `ON_BLOCK` (Blocker), `AT_BREAK_SHIELD`, `ON_ATTACK_FROM_HAND` (Revolution Change), `ON_OTHER_ENTER`
*   **新規**: `ON_SHIELD_ADD`, `ON_CAST_SPELL`

### 2.5 実装上の不整合・未完了項目 (Identified Implementation Inconsistencies)
現在、以下の不整合が確認されており、将来的な修正対象として記録されています。

1.  **一部トリガーのロジック未実装**
    *   なし（`ON_SHIELD_ADD`および`ON_CAST_SPELL`は実装済み）

（現在、主要な不整合は解消されました。）

### 2.6 現在の懸念事項と既知の不具合 (Current Concerns and Known Issues)

以下の不具合および制限事項が確認されており、次期開発フェーズでの修正が必要です。

1.  **モジュール終了時のエラー**
    *   **修正済み**: バッチ推論コールバックのクリーンアップ処理を追加したことにより、Pythonプロセス終了時の `Fatal Python error` は解消されました。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

今後は、システムの堅牢性を高めるリファクタリング、ユーザビリティの向上、およびAIモデルのアーキテクチャ刷新（Transformer）を目指します。

### 3.0 [Priority: Immediate] Refactoring and Stabilization (基盤安定化とリファクタリング)

開発効率と信頼性を向上させるため、以下の修正と機能拡張を最優先で実施します。

1.  **汎用コストシステムのエンジン統合 (Cost System Integration) [完了]**
    *   **現状**: `CostPaymentSystem` を `ActionGenerator` (MainPhaseStrategy) および `EffectResolver` に統合しました。
    *   **検証**: `tests/test_cost_payment_integration.py` にて正常動作を確認済み。

2.  **変数リンク機能の不具合修正 (Variable Linking Fix) [完了]**
    *   **現状**: `GenericCardSystem` におけるアクション間のコンテキスト伝播の修正完了。
    *   **検証**: `tests/verify_atomic_versatility.py` が全テストケースで通過することを確認済み。

3.  **トリガーロジックの補完 (Trigger Logic Completion) [完了]**
    *   **現状**: `ON_SHIELD_ADD` および `ON_CAST_SPELL` のトリガータイプを `bindings.cpp` に追加し、`MoveCardHandler` および `EffectResolver` に発火ロジックを実装しました。
    *   **検証**: 変数リンク検証時にトリガーの列挙子が正しく機能していることを確認済み。

### 3.1 [Priority: High] User Requested Enhancements (ユーザー要望対応 - 残件)

直近のフィードバックに基づく残存タスク。

1.  **GUI/Editor 機能拡張 [完了]**
    *   **リアクション編集の最適化**: `ReactionWidget` を更新し、能力タイプ（Ninja Strike, Strike Back等）に応じたUIの動的切り替えと、不必要なフィールドの隠蔽を実装しました。
    *   **テキスト生成の拡充 (New)**: `CardTextGenerator` を更新し、数値範囲（「1〜5の数字を選ぶ」）や任意選択（「〜まで選ぶ」「そうしてもよい」）の日本語生成ロジックを強化しました。

### 3.2 [Priority: Medium] Phase 4: アーキテクチャ刷新 (Architecture Update)

1.  **Transformer (Linear Attention) 導入 [完了]**
    *   **目的**: 盤面のカード枚数が可変であるTCGの特性に合わせ、固定長入力のResNetから、可変長入力を扱えるAttention機構へ移行する。
    *   **計画**: `NetworkV2` として、PyTorchでのモデル定義と、C++側のテンソル変換ロジック（`TensorConverter`）の書き換えを行う。
    *   *ステータス: 実装完了・検証済み*。
        *   `dm_toolkit/training/network_v2.py` に `LinearAttention` およびTransformerベースの `NetworkV2` を実装。
        *   `src/ai/encoders/tensor_converter.cpp` に `convert_to_sequence` を実装し、カードIDのトークン化とシーケンス変換をC++エンジン側でサポートしました。
        *   単体テスト (`tests/python/training/test_network_v2.py`, `tests/verify_phase3_4.py`) により、可変長シーケンス処理の動作を確認済み。

### 3.3 [Priority: Low] Phase 5: エディタ機能の完成形 (Editor Polish)

AI開発と並行して、エディタの残存課題を解消します。

1.  **詳細なバリデーション**
    *   互換性のないアクションやトリガーの組み合わせに対する警告表示。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし）

---

## 5. イベント駆動型トリガーシステム実装詳細 (Event-Driven Trigger System Implementation Specification)

現状の `EffectResolver` 等に散在するハードコードされたトリガー判定ロジック（ポーリング型/Hook型）を、汎用的かつ拡張可能なイベント駆動型アーキテクチャに刷新するための技術詳細です。

### 5.1 目的 (Objective)
*   **ハードコードの排除**: 新しいトリガー条件を追加するたびにC++エンジン（`EffectResolver` 等）を修正する必要をなくす。
*   **複雑な条件のサポート**: 「相手が呪文を唱えた時」「他のクリーチャーが破壊された時」などの他者・他領域のイベントに対するトリガーを統一的に扱う。
*   **データ駆動**: JSON定義のみで新しい誘発型能力を実装可能にする。

### 5.2 データ構造 (Data Structures)

#### 5.2.1 ゲームイベント定義 (Game Event Definition)
すべてのゲーム内アクション（状態変化）はイベントとして発行されます。

```cpp
// src/core/game_event.hpp (New)

namespace dm::core {

    enum class GameEventType {
        ZONE_CHANGE,        // カードの移動 (Play, Destroy, ManaCharge, Draw)
        ATTACK_INITIATE,    // 攻撃開始
        BLOCK_DECLARED,     // ブロック成立
        CARD_TAPPED,        // カードがタップされた
        PHASE_CHANGED,      // フェーズ遷移
        TURN_START,         // ターン開始
        TURN_END,           // ターン終了
        MANA_PAID,          // マナ支払い
        SPELL_CAST,         // 呪文詠唱完了
        SHIELD_BROKEN,      // シールドブレイク
        POWER_MODIFIED      // パワー変更（継続的効果の再計算トリガー用）
    };

    struct GameEvent {
        GameEventType type;
        int source_id;          // イベント発生源のCardInstance ID
        int target_id;          // 対象のID（攻撃対象、ブロック対象など）
        PlayerID player_id;     // 行動主
        std::map<std::string, int> context; // 追加データ（支払ったコスト、移動元ゾーンIDなど）

        // Context Helper
        void set_context(const std::string& key, int value) { context[key] = value; }
        int get_context(const std::string& key, int default_val = 0) const {
            auto it = context.find(key);
            return (it != context.end()) ? it->second : default_val;
        }
    };
}
```

#### 5.2.2 トリガー定義 (Trigger Definition)
JSON定義上の `TriggerType` を拡張し、より詳細なイベントフィルタリングを可能にします。

```cpp
// src/core/card_json_types.hpp (Extension)

struct EventFilterDef {
    std::vector<std::string> event_types; // "ZONE_CHANGE", "ATTACK_INITIATE" etc.
    FilterDef source_filter;    // イベント発生源が条件を満たすか（例: ドラゴンが破壊された時）
    FilterDef context_filter;   // 文脈条件（例: 手札から捨てられた時）
    bool run_once_per_turn = false;
};

// Existing TriggerType will be mapped to presets of EventFilterDef
// e.g., ON_PLAY -> { event_types: ["ZONE_CHANGE"], source_filter: { zones: ["BATTLE_ZONE"], prev_zone: ["HAND", "MANA_ZONE", ...] } }
```

### 5.3 アーキテクチャ (Architecture)

#### 5.3.1 TriggerManager (Observer Pattern)
`GameState` またはシングルトンとして `TriggerManager` を実装し、イベントの監視と発火を管理します。

```cpp
// src/engine/systems/trigger_system.hpp

class TriggerManager {
public:
    // イベント発行（各ActionHandlerから呼ばれる）
    void dispatch(GameState& state, const GameEvent& event, const CardDB& card_db);

    // 監視対象の登録（不要？ -> 毎回全カード走査は重いため、ZoneごとのObserverリストを持つか、
    // あるいは「現在バトルゾーン/マナゾーンにあるカード」のみを走査対象とする）
    // 現状の規模なら、Event発生時に全Active Zone（Battle, Mana, Grave, Handの一部）を走査しても十分高速。
};
```

### 5.4 実装ステップ (Implementation Steps)

1.  **インフラ整備**:
    *   `src/core/game_event.hpp` の作成。
    *   `src/engine/systems/trigger_system.hpp` の作成。

2.  **イベント発行の実装 (Event Emission)**:
    *   `MoveCardHandler`, `TapHandler`, `AttackHandler` 等の主要ハンドラおよび `EffectResolver` 内のハードコード部分に `TriggerManager::dispatch(event)` を埋め込む。
    *   例: `GenericCardSystem` でカードが移動する直前/直後に `ZONE_CHANGE` イベントを発行。

3.  **互換性レイヤーの実装**:
    *   既存の `TriggerType` (ON_PLAY, ON_DESTROY等) を `EventFilterDef` に変換するロジック、または `TriggerManager` 内で従来の `TriggerType` をイベントとして解釈するロジックを実装。

4.  **移行と検証**:
    *   `ON_DESTROY` ロジックを `TriggerManager` 経由に切り替え、既存の破壊時効果（Piggyback）が動作するか検証。
    *   `ON_OPPONENT_DRAW` などの新規トリガーをイベント駆動で実装し、動作確認。

---

## 今後の課題 (Future Issues)

### 1. Handlerの更なる堅牢化
`DestroyHandler` 等のテストにおいて、特定条件下（例：テスト環境でのSource IDのオーナー解決不整合）で意図した挙動にならないケースが観測されています。実環境では正常動作する可能性が高いですが、テスト基盤とエンジン側の疎結合性を高め、より堅牢なユニットテスト体制を構築する必要があります。

### 2. 変数リンクの適用範囲拡大
**[完了]** 主要なハンドラ（Draw, Destroy, Shield, Count, BreakShield）に加え、`TapHandler`, `UntapHandler`, `ManaChargeHandler` においても `GenericCardSystem` を介した変数リンク（`input_value_key`）のサポートが確認されました。これにより、ほぼ全てのアクションで動的な数値指定が可能です。

### 3.4 [Priority: Future] Phase 6: 将来的な拡張性・汎用性向上 (Future Scalability)

1.  **イベント駆動型トリガーシステム (Event-Driven Trigger System)**
    *   （詳細仕様は Section 5 を参照）
2.  **AI入力特徴量の動的構成 (Dynamic AI Input Feature Configuration)**
3.  **完全な再現性を持つリプレイシステム (Fully Reproducible Replay System)**

### 3.5 [Priority: Future] Action & Filter Generalization (アクションとフィルタの汎用化)

現在のハードコードされた多くのアクションタイプやフィルタリングロジックを統合し、データ駆動で柔軟なカード定義を可能にするためのリファクタリング計画です。

1.  **カード移動アクションの完全統合 (`MOVE_CARD` の汎用化)**
    *   **目的**: `DESTROY_CARD`, `RETURN_TO_HAND`, `SEND_TO_MANA`, `SEND_TO_SHIELD` などの個別の移動アクションを、単一の `MOVE_CARD` アクションに統合します。
    *   **仕様**:
        *   必須引数 `destination`: `GRAVEYARD`, `HAND`, `MANA`, `SHIELD`, `DECK`
        *   オプション `position`: `TOP`, `BOTTOM`, `SHUFFLE`, `RANDOM`
        *   オプション `face_down`: `true` / `false`
    *   **効果**: エディタのUI簡素化と、将来的な新しいゾーン移動への対応コスト削減。

2.  **修正適用アクションの統合 (`APPLY_MODIFIER`)**
    *   **目的**: `GIVE_BLOCKER`, `BUFF_POWER` などのバフ・デバフ系アクションを `APPLY_MODIFIER` に統合します。
    *   **仕様**:
        *   `duration`: `UNTIL_END_OF_TURN`, `PERMANENT`, `WHILE_ON_BATTLEZONE`
        *   `modifier` (Object): `power_add` (int), `add_keywords` (List), `remove_keywords` (List)
    *   **効果**: 複数の効果（例：パワー+3000してW・ブレイカーを与える）を単一のアクション定義で記述可能にする。

3.  **イベント監視型トリガー (`OBSERVER_TRIGGER`)**
    *   （実装詳細については **Section 5: イベント駆動型トリガーシステム実装詳細** に統合されました）

4.  **動的参照フィルタ (`Source Reference`)**
    *   **目的**: 固定値（パワー5000以下）だけでなく、動的な値（自身のパワー以下、攻撃対象と同じ種族）によるフィルタリングを可能にします。
    *   **仕様**:
        *   `FilterDef` 内に `*_ref` フィールドを追加（例: `max_power_ref: SELF_POWER`, `race_match_ref: TARGET_RACE`）。
    *   **効果**: 状況依存の除去やサーチ効果の実装。

5.  **汎用計算アクション (`CALCULATE` / `MATH_OP`)**
    *   **目的**: 変数間の計算処理（加算、乗算など）をアクションとして定義可能にします。
    *   **仕様**:
        *   `operation`: `ADD`, `MULTIPLY`, `DIVIDE`, `MIN`, `MAX`
        *   `operand`: 固定値または変数キー
        *   `scale`: 係数
    *   **効果**: 「シールド1枚につき+1000」のような動的な数値補正ロジックをコード変更なしで実装可能にする。

## Kaggle クラウドデータ収集システム 運用マニュアル

（内容は変更なし）
