# 要件定義書 00: ステータスと要件の概要

## 1. プロジェクト概要
**目標:** C++エンジンとPythonバインディングを使用したデュエル・マスターズAI/エージェントシステムの開発。
**現在のフェーズ:** フェーズ5 (汎用エンジン機能と特徴量抽出)
**焦点:** 堅牢性、検証可能性、リスク管理。特に、エンジンの「原子アクション分解」を行い、複雑な効果処理をシンプルかつデータ駆動で扱えるようにリファクタリングする。

## 2. 現在のステータス
*   **エンジン:** `GameState`、`ActionGenerator`、`EffectResolver`を備えたC++20コア。汎用的な効果解決（Effect Buffer, Stack Zone）を実装済み。
*   **進捗:** 基本的な対戦、コスト軽減、ハイパーエナジー、革命チェンジの実装が完了。
*   **Pythonバインディング:** `pybind11`統合 (`dm_ai_module`)。
*   **AI:** AlphaZeroスタイルのMCTS (`ParallelRunner`) + PyTorch学習 (`train_simple.py`)。
*   **GUI:** Python `tkinter` ベース (`app.py`) および `PyQt6` ベース (`card_editor.py`)。
*   **データ:** JSONベースのカード定義 (`data/cards.json`)。

## 3. 現在の開発段階 (Current Development Stage)
**ステータス:** **エンジンフローのリファクタリングとメタカウンター実装**

原子アクション分解（プレイ、バトル、マナチャージ）が完了し、現在はスタックとゲートキーパーの統合、およびメタカウンター踏み倒し機能の実装フェーズに移行しています。

## 4. アクティブな要件とタスク (Uncompleted Tasks)

### 4.1. エンジンリファクタリング：原子アクション分解
以下の提案に基づき、エンジンのコアアクション処理を分解・再定義します。これにより、コードの重複を排除し、新しいメカニクス（G・ゼロ、シールド焼却など）の実装を容易にします。

#### 1. 汎用アクションの完全定義
*   **A. MOVE_CARD (完全共通化) (一部完了):**
    *   アクション: `MOVE_CARD(source_zone, dest_zone, card_instance_id)`
    *   破壊、バウンス、ドロー、シールド化を全て統合し、「ゾーンを離れた時/置かれた時」のフック処理を一元化します。
    *   *現状:* マナチャージのみ先行して実装済み。
*   **B. TAP_CARD / UNTAP_CARD (分離):**
    *   アクション: `TAP_CARD(target_instance_id)`
    *   攻撃、ブロック、支払いに含まれるタップ処理を分離し、効果によるタップ/アンタップを容易にします。
*   **C. SHUFFLE_DECK:**
    *   サーチや回復効果のために単独アクション化します。
*   **D. BATTLE_CREATURES:**
    *   ブレイク要素を排除した純粋なバトル処理として定義します。

#### 2. その他機能拡張のためのアクション (実装完了)
*   **APPLY_MODIFIER (継続的効果):**
    *   アクション: `APPLY_MODIFIER(modifier_def, duration, target_filter)`
    *   コスト軽減やパワー修正を履歴に残るアクションとして定義し、AIが理由を追跡可能にします。
*   **REVEAL_CARDS (公開):**
    *   アクション: `REVEAL_CARDS(zone, indices, visibility_scope)`
    *   「見る」処理を明確化し、AIの観測情報（Observation）更新ロジックと連動させます。
*   **REGISTER_DELAYED_EFFECT (遅延誘発):**
    *   アクション: `REGISTER_DELAYED_EFFECT(trigger_condition, effect_def)`
    *   「ターン終了時」などの効果予約をデータ駆動（JSON）で行えるようにします。
*   **RESET_INSTANCE (初期化):**
    *   アクション: `RESET_INSTANCE(instance_id)`
    *   ゾーン移動時の召喚酔いや修正値のリセットを明示的なアクションとして分離します。

### 4.2. GUI & カード効果の拡張
*   **カード作成補助 (日本語化 & 視覚化):**
    *   **日本語化:** GUI上のキーワード能力や汎用能力の選択を日本語で行えるようにする。
    *   **視覚的ビルダ:** トリガー条件、対象ゾーン、処理（見る、表向きにする、手札に戻す、参照など）、対象数、対象条件、コスト軽減などを視覚的に選択・組み合わせて複雑な効果を実装できる機能を追加する。

### 4.3. 将来の機能バックログ (ユーザー要望)
*   **システム & エンジン:**
    *   ドロー監視、墓地ロジックの拡張。
*   **カードメカニクス:**
    *   ジャストダイバー、代替コスト、メテオバーン、NEO進化、ニンジャ・ストライク、ブロック不可、全体除去、攻撃制限、アンチチート、マナ回収、リアニメイト、モード効果。

### 4.4. 実装計画：メタカウンターとエンジンコア拡張

#### 1. エンジンフローのリファクタリング (スタックとゲートキーパー)
- **EffectResolver の拡張**
    - [完了] `src/engine/effects/effect_resolver.cpp` の `resolve_play_from_stack` を修正し、`SpawnSource` 引数を受け取れるようにしました。
    - **ゲートキーパー (Gatekeeper) ロジックの実装**:
        - [完了] カードがバトルゾーンに出る直前に、「移動先決定ロジック」を挟みます（SpawnSourceによる分岐準備完了）。
    - `ActionType::PLAY_CARD_INTERNAL` を処理するケースを追加し、`resolve_play_from_stack` へ委譲します。
- **ActionGenerator の更新**
    - `src/engine/action_gen/action_generator.cpp` を更新し、`EffectType::INTERNAL_PLAY` や `META_COUNTER` が `pending_effects` にある場合、`PLAY_CARD_INTERNAL` アクションを生成するようにします。

#### 2. 既存メカニクスのスタック移行
- **直接 `push_back` の廃止**
    - 以下の箇所で、直接バトルゾーンに追加している処理を廃止し、代わりに `PendingEffect` (Type: `INTERNAL_PLAY`) を積む形に変更します。
        - `src/engine/card_system/generic_card_system.cpp`: メクレイド (MEKRAID)、バッファからのプレイ
        - `src/engine/effects/effect_resolver.cpp`: S・トリガー (SHIELD_TRIGGER) の解決処理

#### 3. カウンター踏み倒し機能の実装
- **ターン終了時のチェック**
    - `src/engine/flow/phase_manager.cpp` の `next_phase` にロジックを追加します。
- **解決フロー**
    - 積まれた `META_COUNTER` は `ActionGenerator` によってアクション化され、スタック経由（`PLAY_CARD_INTERNAL`）で解決されます。

#### 4. 検証
- **テスト作成**
    - `tests/test_meta_counter.py` を作成し検証します。
- **Pre-commit**
    - 全テストの通過を確認し、コードフォーマット等を整えます。

### 4.5. 実装方針まとめ：動的値参照とカードエディタの拡張 (実装完了)

#### 1. 概要
「マナゾーンの文明数分ドロー」や「ドローした枚数分コスト軽減」といった複雑な効果を実現するため、**「レジスタ（変数）連携システム」**を導入しました。

#### 2. C++ エンジン拡張 (src/core/)
*   **2.1 データ構造の変更 (ActionDef の拡張) [完了]**
    *   `input_value_key`, `output_value_key` を追加。

*   **2.2 新規アクションタイプ (EffectActionType) [完了]**
    *   `COUNT_CARDS`
    *   `GET_GAME_STAT`
    *   `APPLY_MODIFIER`
    *   `REVEAL_CARDS`
    *   `REGISTER_DELAYED_EFFECT`
    *   `RESET_INSTANCE`

*   **2.3 統計情報の拡充 (TurnStats) [完了]**
    *   `cards_drawn_this_turn`
    *   `cards_discarded_this_turn`
    *   `creatures_played_this_turn`
    *   `spells_cast_this_turn`

*   **2.4 ロジックの実装 (EffectResolver) [完了]**
    *   `resolve_effect` に `execution_context` を実装し、変数連携をサポート。

#### 3. カードエディタ (GUI) の改良 (python/gui/card_editor.py) [一部完了]
*   **入力UI:** アクション詳細設定に `Input Key` と `Output Key` の入力フィールドを追加。
*   **今後の課題:** 「スマート化」UI（直感的な参照設定）は未実装だが、JSONベースでの編集は可能になった。

#### 4. テストと検証 [完了]
*   `tests/test_dynamic_values.py`: `COUNT_CARDS` と `GET_GAME_STAT` の動作確認済み。

## 5. 既知の問題 / リスク
*   **複雑な効果:** 複数ステップの効果 (探索、シールドトリガーの選択) はC++での堅牢な処理が必要です。
*   **メモリ使用量:** `verify_performance.py` でのシミュレーション回数が多いと、メモリアロケーションエラー (`std::bad_alloc`) が発生する可能性があります。
*   **データの一貫性:** `data/cards.csv` はレガシーであり、`data/cards.json` を使用する必要があります。

## 6. 今後の実装計画 (Future Implementation Plan)

### [PLAN-003] Transformerアーキテクチャの実装
*   **概要:** 現在のCNN/ResNetモデルから、Transformer (Self-Attention) モデルへの移行。
*   **目的:** 盤面の複雑な相互作用を捉え、より戦略的な深さを持つAI「自筆進化エコシステム」の基盤とする。

### [PLAN-004] PBT (Population Based Training) 基盤の構築
*   **概要:** 数千〜数万試合規模の並列対戦を実行し、エージェントを進化させるパイプラインの構築。
*   **目的:** メタゲームの変遷に適応する堅牢なAIを作成する。

## 7. 開発規約 (Development Conventions)
*   **コミットメッセージ:** 日本語で記述する。 (`[Type] Subject`)
