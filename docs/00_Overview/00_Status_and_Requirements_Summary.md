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
*   **汎用コストシステム（導入済）**: `CostPaymentSystem` を実装し、能動的コスト軽減（ハイパーエナジー等）の計算基盤を確立しました（アクション統合は進行中）。
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
    *   **安定性**: プロパティインスペクタのクラッシュバグを修正済み。
    *   **不整合修正**: 革命チェンジのデータ構造、文明指定キーの不整合を解消済み。
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
*   **Phase 4 アーキテクチャ**: `NetworkV2` (Transformer/Linear Attention) の実装と単体テストによる動作検証を完了しました。

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

### 2.5 実装上の不整合・未完了項目 (Identified Implementation Inconsistencies)
現在、以下の不整合が確認されており、将来的な修正対象として記録されています。

1.  **一部トリガーのロジック未実装**
    *   `ON_SHIELD_ADD`: Enum定義済みですが、`MoveCardHandler` 等でのフック処理がコメントアウト状態または不完全です（リアクションシステムでは一部参照あり）。
    *   `ON_CAST_SPELL`: Enum定義済みですが、`CastSpellHandler` 等での発火ロジックが明示的に見当たりません。

（現在、主要な不整合は解消されました。上記はマイナーな残存課題です。）

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

今後は、システムの堅牢性を高めるリファクタリング、ユーザビリティの向上、およびAIモデルのアーキテクチャ刷新（Transformer）を目指します。

### 3.0 [Priority: Immediate] Refactoring and Stabilization (基盤安定化とリファクタリング)

開発効率と信頼性を向上させるため、以下の修正と機能拡張を最優先で実施します。

1.  **汎用コストシステムのエンジン統合 (Cost System Integration)**
    *   **現状**: `CostPaymentSystem` クラスとデータ構造の実装は完了 (`test_cost_payment_structs.py` で検証済)。
    *   **タスク**: `ActionGenerator` および `ManaSystem` (または `EffectResolver`) を修正し、実際のゲームループ内で能動的コスト軽減（クリーチャーのタップ等）を伴うプレイを行えるようにする。

### 3.1 [Priority: High] User Requested Enhancements (ユーザー要望対応 - 残件)

直近のフィードバックに基づく残存タスク。

1.  **GUI/Editor 機能拡張**
    *   **リアクション編集の最適化**
        *   リアクション能力のウィジェットは、将来的にEffect編集画面等、より適切なコンテキストで編集できるように検討する。

### 3.2 [Priority: Medium] Phase 4: アーキテクチャ刷新 (Architecture Update)

1.  **Transformer (Linear Attention) 導入**
    *   **目的**: 盤面のカード枚数が可変であるTCGの特性に合わせ、固定長入力のResNetから、可変長入力を扱えるAttention機構へ移行する。
    *   **計画**: `NetworkV2` として、PyTorchでのモデル定義と、C++側のテンソル変換ロジック（`TensorConverter`）の書き換えを行う。
    *   *ステータス: 実装完了・検証済み*。
        *   `dm_toolkit/training/network_v2.py` に `LinearAttention` およびTransformerベースの `NetworkV2` を実装。
        *   単体テスト (`tests/python/training/test_network_v2.py`) により、可変長シーケンス処理とマスキングロジックの動作を確認済み。

### 3.3 [Priority: Low] Phase 5: エディタ機能の完成形 (Editor Polish)

AI開発と並行して、エディタの残存課題を解消します。

1.  **詳細なバリデーション**
    *   互換性のないアクションやトリガーの組み合わせに対する警告表示。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

本システムは、マナ以外のリソース（手札、バトルゾーンのクリーチャー、シールドなど）によるコスト支払い、およびコスト軽減や追加コストを統一的に扱うための仕様定義です。
特に、支払うリソースの量に応じてコストが変動する（スケーリングする）ケース（例：ハイパーエナジーの複数体タップ）に対応します。

### 4.1 概要 (Overview)
カードの使用や能力の発動に必要な「代償（コスト）」を以下の3要素に分類し、JSONデータ構造として定義可能にします。

1.  **代替コスト (Alternative Cost)**: マナコストを完全に置き換える（例：G・ゼロ）。
2.  **追加コスト (Additional Cost)**: プレイに追加で支払う（例：キッカー）。
3.  **能動的コスト軽減 (Active Cost Reduction)**: 任意のリソースを支払うことで、1単位ごとにマナコストを軽減する（例：ハイパーエナジー）。

### 4.2 データ構造 (Data Structure Specifications) - 実装済み

`CostType`, `CostDef`, `CostReductionDef` は `src/core/card_json_types.hpp` に実装済みです。
また、計算ロジック `CostPaymentSystem` も実装済みです。

### 4.3 処理ロジック (Processing Logic)

エンジン (`CostPaymentSystem`) は以下のフローで処理を行います。

1.  **支払い可能回数の計算**:
    *   戦場の対象カード数などをカウントし、最大何回まで軽減を適用可能か判定します。(実装済)
2.  **アクション生成** (未実装 - Next Step):
    *   軽減可能な場合、「コストを支払ってプレイ (Variable Payment)」アクションを生成します。
3.  **解決プロセス** (未実装 - Next Step):
    *   **Step 1**: ユーザーが軽減のための支払いを実行（例：クリーチャーをタップ）。
    *   **Step 2**: 残りのマナコストを計算し、マナ支払い判定を行います。
    *   **Step 3**: 最終的なコスト（リソース＋マナ）を支払い、カードをプレイします。

### 4.4 定義例: スケーリングハイパーエナジー (Hyper Energy Example)
（定義例は変更なし）

---

## 今後の課題 (Future Issues)

### 1. Handlerの更なる堅牢化
`DestroyHandler` 等のテストにおいて、特定条件下（例：テスト環境でのSource IDのオーナー解決不整合）で意図した挙動にならないケースが観測されています。実環境では正常動作する可能性が高いですが、テスト基盤とエンジン側の疎結合性を高め、より堅牢なユニットテスト体制を構築する必要があります。

### 2. 変数リンクの適用範囲拡大
現在、主要なハンドラ（Draw, Destroy, Shield, Count）での変数リンク対応が完了しましたが、他のすべてのアクションハンドラ（例：ManaCharge, Tap/Untap）にも同様の仕組みを波及させ、あらゆる数値を動的に制御可能にすることが推奨されます。

### 3.4 [Priority: Future] Phase 6: 将来的な拡張性・汎用性向上 (Future Scalability)

要件定義書00に含まれていないが、エンジンの長期的な保守性と拡張性を担保するために以下の実装を提案します。これらは重要度順に記載されています。

1.  **イベント駆動型トリガーシステム (Event-Driven Trigger System)**
    *   **現状**: `ON_PLAY` (cip), `ON_ATTACK`, `ON_DESTROY` などのトリガーは、フェーズ処理(`PhaseManager`)や解決ロジック(`EffectResolver`)の中でハードコードされた分岐として実装されています。「革命チェンジ」や「ストライク・バック」も専用のロジックとして追加されています。
    *   **提案**: オブザーバーパターンを用いた汎用イベントバスを導入します。
    *   **効果**: 新しいキーワード能力や誘発型能力を追加する際、C++のコアロジック（フロー制御）を修正する必要がなくなり、データ定義のみで完結する範囲が大幅に広がります。

2.  **AI入力特徴量の動的構成 (Dynamic AI Input Feature Configuration)**
    *   **現状**: ニューラルネットワークへの入力テンソル（特徴量）の定義はC++コード内で固定（約200次元）されています。
    *   **提案**: 特徴量の構成要素（自分の手札枚数、相手のマナ文明分布など）をJSON設定ファイルで定義し、実行時に動的にテンソルを構築する仕組みを導入します。

3.  **完全な再現性を持つリプレイシステム (Fully Reproducible Replay System)**
    *   **現状**: エラー発生時のデバッグはテキストログに依存しています。
    *   **提案**: 初期シード値、プレイヤーデッキ、およびアクションIDの列のみを記録した軽量なバイナリ形式のリプレイファイルを定義し、GUI上で任意の時点まで状態を復元・再生できるビューアを実装します。

## Kaggle クラウドデータ収集システム 運用マニュアル

（内容は変更なし）
