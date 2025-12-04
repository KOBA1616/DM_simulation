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

#### 2. その他機能拡張のためのアクション
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
    - `src/engine/effects/effect_resolver.cpp` の `resolve_play_from_stack` を修正し、`SpawnSource` 引数を受け取れるようにします。
    - **ゲートキーパー (Gatekeeper) ロジックの実装**:
        - カードがバトルゾーンに出る直前に、「移動先決定ロジック」を挟みます。
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

### 4.5. 実装方針まとめ：動的値参照とカードエディタの拡張

#### 1. 概要
「マナゾーンの文明数分ドロー」や「ドローした枚数分コスト軽減」といった複雑な効果を実現するため、**「レジスタ（変数）連携システム」**を導入します。これは、計算を行うアクションと、その結果を利用するアクションを変数（レジスタ）を介して連携させるアーキテクチャです。また、これをユーザーが直感的に操作できるようGUI（カードエディタ）を改良します。

#### 2. C++ エンジン拡張 (src/core/)
*   **2.1 データ構造の変更 (ActionDef の拡張)**
    *   `src/core/card_json_types.hpp` の `ActionDef` 構造体に、アクション間での値の受け渡しを行うためのフィールドを追加します。
    *   `input_value_key (std::string)`: 値の入力元となる変数名。設定されている場合、固定値 `value1` の代わりにこの変数の値を使用します。
    *   `output_value_key (std::string)`: アクションの結果（カウント数、移動枚数など）の出力先となる変数名。

*   **2.2 新規アクションタイプ (EffectActionType)**
    *   値を動的に生成・計算するための「計算系アクション」を追加します。
    *   `COUNT_CARDS`
        *   機能: 指定されたフィルター（ゾーン、文明、種族など）に合致するカードの枚数を数える。
        *   出力: カウント結果を `output_value_key` に保存。
    *   `GET_GAME_STAT`
        *   機能: ゲーム内の統計情報を取得する。
        *   パラメータ: 取得したい統計のキー（例: "drawn_cards_this_turn", "discarded_cards_this_turn"）。
        *   出力: 統計値を `output_value_key` に保存。
    *   `MATH_OP` (今回はスコープ外だが設計に含める)
        *   機能: 変数同士の加算・乗算など。

*   **2.3 統計情報の拡充 (TurnStats)**
    *   動的参照に対応するため、`GameState` 内の `TurnStats` で追跡する項目を増やします。
    *   `cards_drawn_this_turn` (このターン引いたカードの枚数)
    *   `cards_discarded_this_turn` (このターン捨てたカードの枚数)
    *   `creatures_played_this_turn` (このターン出したクリーチャー数)
    *   `spells_cast_this_turn` (このターン唱えた呪文数)

*   **2.4 ロジックの実装 (EffectResolver)**
    *   `resolve_effect` メソッド内に、効果処理中のみ有効な一時変数マップ (`std::map<std::string, int> execution_context`) を保持します。
    *   各アクション実行前に `input_value_key` を確認し、マップから値を解決してアクションに適用します。
    *   アクション実行後に `output_value_key` があれば、結果をマップに保存します。

#### 3. カードエディタ (GUI) の改良 (python/gui/card_editor.py)
ユーザーには「変数を定義して...」といった複雑な操作を意識させず、直感的に操作できるUIを提供します。

*   **3.1 ゾーン選択の整理**
    *   仕様: ユーザーが選択可能なゾーンから、内部処理用ゾーン（STACK_ZONE, EFFECT_BUFFER 等）を完全に除外し、以下の6つのみを表示・選択可能にします。
        *   DECK, HAND, MANA_ZONE, BATTLE_ZONE, GRAVEYARD, SHIELD_ZONE

*   **3.2 アクション設定の「スマート化」**
    *   **ドローの自動化:** アクションタイプ「ドロー (DRAW_CARD)」を選択した際、対象ゾーン (DECK) とプレイヤー (SELF) を自動設定し、変更不可（またはデフォルト設定）にします。
    *   **数値入力の拡張:** 「値 (Value)」の入力欄を、以下の3モード切替式コンポーネントにアップグレードします。
        *   **固定値:** 従来の数値入力（SpinBox）。
        *   **参照 (Reference):** 他のゾーンや状態を参照するモード。
            *   ドロップダウンで参照元を選択（例：「マナゾーンの文明数」「このターンのドロー枚数」）。
            *   必要に応じてフィルター条件を設定（例：文明＝火）。
        *   **変数 (Variable):** 以前のアクションで出力された変数名を使用（上級者向け）。

*   **3.3 裏側でのJSON生成（コンパイラ機能）**
    *   GUI上では「マナの文明数分ドロー」という1つの設定に見えますが、保存時に自動的に以下の2つのアクションに展開してJSONに書き込みます。
        1.  `COUNT_CARDS` (対象: マナ, 出力: temp_var)
        2.  `DRAW_CARD` (入力: temp_var)
    *   読み込み時はこのパターンを解析し、GUI上で元の「参照モード」として表示を復元します。

#### 4. テストと検証
*   新しい `TurnStats` が正しくカウントされているかを確認するユニットテスト。
*   「マナ数分ドロー」などの動的効果を持つカードを定義し、期待通りに動作するかを確認するテスト。
*   GUIでこれらの設定が正しく行え、JSONとして正しく保存・ロードできるかの検証。

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
