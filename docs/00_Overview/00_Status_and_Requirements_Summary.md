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
*   **アクションシステム**: `IActionHandler` による完全なモジュラー構造。
    *   **カード移動の統一**: `MOVE_CARD` アクションを導入し、DRAW, ADD_MANA, DESTROYなどの専用アクションを汎用移動ロジックで表現可能にしました。
    *   **条件判定の汎用化**: `COMPARE_STAT` 条件を導入し、「手札枚数がX以上」などの数値比較条件を自由に記述可能にしました。
    *   **変数リンクの完全サポート**: 各アクションハンドラ（Draw, Destroy, Shield等）において、`input_value_key` による入力値の参照と `output_value_key` による実行結果の書き込み（例：破壊した数の出力）をサポートしました。
    *   **モード選択機能**: `SELECT_OPTION` アクションを導入し、「選択肢から選ぶ」効果の実装と、ネストされたアクションのGUI編集に対応しました。
*   **高速シミュレーション**: OpenMPによる並列化と最適化されたメモリ管理により、秒間数千〜数万試合の自己対戦が可能。
*   **不完全情報探索 (PIMC)**: 推定された世界でのモンテカルロ探索（Determinization）を並列実行可能。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.2**: 3ペイン構成（ツリー/プロパティ/プレビュー）に加え、ツインパクトや革命チェンジのロジック自動生成機能を統合。
*   **機能**: JSONデータの視覚的編集、ロジックツリー、変数リンク、テキスト自動生成、デッキビルダー、シナリオエディタ。
*   **検証済み**: 生成されたJSONデータはエンジンで即座に読み込み可能。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **AlphaZero Pipeline**: データ収集 -> 学習 -> 評価 の完全自動ループが稼働中。
*   **推論エンジン**: 相手デッキタイプ推定 (`DeckClassifier`) と手札確率推定 (`HandEstimator`) を実装済み。
*   **自己進化**: 遺伝的アルゴリズムによるデッキ改良ロジック (`DeckEvolution`) がC++コアに統合され、高速に動作します。

### 2.4 現在確認されている実装上の不整合 (Identified Implementation Inconsistencies)
現在、以下の不整合が確認されており、将来的な修正対象として記録されています。

1.  **革命チェンジのデータ構造不整合**
    *   **Card Editor**: 革命チェンジのロジックを「Effect内のAction (`REVOLUTION_CHANGE`)」として保存します。
    *   **Engine (`JsonLoader`)**: 革命チェンジの宣言条件として、Card定義のルートレベルにある `revolution_change_condition` (FilterDef) を参照します。
    *   **影響**: エディタで作成した革命チェンジ持ちカードが、エンジン側で正しく「革命チェンジ持ち」として認識されるものの、宣言時の条件フィルタ（種族・コスト指定など）が反映されない可能性があります。

2.  **C++ コンパイル警告 (ConditionDef)**
    *   `CostHandler`, `ShieldHandler`, `SearchHandler` 等において、`ConditionDef` のブレース初期化リストが構造体のフィールド更新（`stat_key`等の追加）に追従しておらず、多くの `missing initializer` 警告が発生しています。

3.  **Atomic Action テストの失敗**
    *   `tests/python/test_new_actions.py` 内の `test_cast_spell_action` が失敗します。
    *   **原因**: テストコードが `CAST_SPELL` アクションを単体で解決しようとしていますが、エンジン側のハンドラが期待するコンテキスト（スタック経由の処理など）と一致していない可能性があります。

4.  **文明指定のキー不整合 (Legacy Support)**
    *   **Editor**: 新規カード作成時に単数形のキー `"civilization"` を使用しています。
    *   **Engine**: 内部構造および推奨フォーマットは複数形の `"civilizations"` です（`JsonLoader` に互換処理が存在するため現状は動作します）。
    *木構造をユーザーが破壊しないようにしたい。
    Traceback (most recent call last):
    File "C:\Users\mediastation36\Documents\DM_simulater\DM_simulation\dm_toolkit\gui\card_editor.py", line 122, in on_selection_changed
    self.inspector.set_selection(None)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
     File "C:\Users\mediastation36\Documents\DM_simulater\DM_simulation\dm_toolkit\gui\editor\property_inspector.py", line 43, in set_selection
    if not index.isValid():
           ^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'isValid'

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

今後は、システムの堅牢性を高めるリファクタリング、ユーザビリティの向上、およびAIモデルのアーキテクチャ刷新（Transformer）を目指します。

### 3.0 [Priority: Immediate] Refactoring and Stabilization (基盤安定化とリファクタリング)

開発効率と信頼性を向上させるため、以下の修正と機能拡張を最優先で実施します。

1.  **基盤リファクタリング (Fundamental Refactoring)**
    *   **パッケージ構造**: `dm_toolkit` をトップレベルパッケージとして確立し、関連するインポートやパス設定を修正済み。
    *   **GUIシミュレーション統合**: GUIの「バッチシミュレーション」機能を `ParallelRunner` バックエンド利用に統一済み (Caution: Large batches may leak memory).

### 3.1 [Priority: High] User Requested Enhancements (ユーザー要望対応 - 残件)

直近のフィードバックに基づく残存タスク。

1.  **GUI/Editor 機能拡張**
    *   **リアクション編集の最適化**
        *   リアクション能力のウィジェットは、将来的にEffect編集画面等、より適切なコンテキストで編集できるように検討する。

### 3.2 [Priority: Medium] Phase 4: アーキテクチャ刷新 (Architecture Update)

1.  **Transformer (Linear Attention) 導入**
    *   **目的**: 盤面のカード枚数が可変であるTCGの特性に合わせ、固定長入力のResNetから、可変長入力を扱えるAttention機構へ移行する。
    *   **計画**: `NetworkV2` として、PyTorchでのモデル定義と、C++側のテンソル変換ロジック（`TensorConverter`）の書き換えを行う。
    *   *ステータス: 着手*。`dm_toolkit/training/network_v2.py` を作成し、O(N)計算量の `LinearAttention` およびTransformerベースの `NetworkV2` クラスの初期実装を行いました。

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

### 4.2 データ構造 (Data Structure Specifications)

既存の `CardDefinition.cost` をベースとしつつ、以下の構造体を追加します。

#### 4.2.1 コスト種別 (CostType)
コストとして支払うリソースの種類を定義します。

```cpp
enum class CostType {
    MANA,               // マナ支払い (追加マナ等)
    TAP_CARD,           // 指定ゾーンのカードをタップ (ハイパーエナジー等)
    SACRIFICE_CARD,     // 指定ゾーンのカードを墓地へ (破壊、手札捨て、シールド焼却)
    RETURN_CARD,        // 指定ゾーンのカードを手札へ戻す (マナ回収コスト等)
    SHIELD_BURN,        // シールドを墓地へ (エスケープ等)
    DISCARD             // 手札を捨てる
};
```

#### 4.2.2 汎用コスト定義 (CostDef)
単一の支払い単位を定義します。

```cpp
struct CostDef {
    CostType type;
    int amount;                     // 支払う量（枚数、マナ数）
    FilterDef filter;               // 対象の条件 (例: 「火のクリーチャー」)
    bool is_optional = false;       // 任意コストかどうか
    std::string cost_id;            // 識別子
};
```

#### 4.2.3 コスト軽減定義 (CostReductionDef)
能動的コスト軽減（スケーリング）を定義します。

```cpp
enum class ReductionType {
    PASSIVE,        // 永続的軽減 (既存: 自分のドラゴン1体につき-1)
    ACTIVE_PAYMENT  // 能動的支払いによる軽減 (新規)
};

struct CostReductionDef {
    ReductionType type;

    // ACTIVE_PAYMENT 用の設定
    CostDef unit_cost;       // 1単位あたりの支払いコスト (例: クリーチャー1体タップ)
    int reduction_amount;    // 1単位あたりの軽減マナ数 (例: 2マナ軽減)
    int max_units = -1;      // 最大適用回数 (-1は無制限)
    int min_mana_cost = 0;   // 軽減後のコスト下限

    std::string name;        // UI表示名
};
```

### 4.3 処理ロジック (Processing Logic)

エンジン (`CostPaymentSystem`) は以下のフローで処理を行います。

1.  **支払い可能回数の計算**:
    *   戦場の対象カード数などをカウントし、最大何回まで軽減を適用可能か判定します。
2.  **アクション生成**:
    *   軽減可能な場合、「コストを支払ってプレイ (Variable Payment)」アクションを生成します。
3.  **解決プロセス**:
    *   **Step 1**: ユーザーが軽減のための支払いを実行（例：クリーチャーをタップ）。
    *   **Step 2**: 残りのマナコストを計算し、マナ支払い判定を行います。
    *   **Step 3**: 最終的なコスト（リソース＋マナ）を支払い、カードをプレイします。

### 4.4 定義例: スケーリングハイパーエナジー (Hyper Energy Example)
「8コストのクリーチャーだが、自分のクリーチャーを1体タップするごとにコストを2軽減できる（下限0）」場合のJSON定義例。

```json
{
  "id": 1000,
  "name": "Scaling Hyper Creature",
  "cost": 8,
  "cost_reductions": [
    {
      "name": "Hyper Energy",
      "type": "ACTIVE_PAYMENT",
      "reduction_amount": 2,
      "min_mana_cost": 0,
      "unit_cost": {
        "type": "TAP_CARD",
        "amount": 1,
        "filter": {
          "zones": ["BATTLE_ZONE"],
          "types": ["CREATURE"],
          "is_tapped": false
        }
      }
    }
  ]
}
```

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
        *   エンジンはゲーム進行中に `Event::ATTACK_DECLARED`, `Event::CARD_MOVED`, `Event::TURN_START` などのイベントを発行します。
        *   カード効果（JSON定義）は特定のイベントに対するリスナーとして登録され、イベント発生時に条件（Condition）を評価してアクションを生成します。
    *   **効果**: 新しいキーワード能力や誘発型能力を追加する際、C++のコアロジック（フロー制御）を修正する必要がなくなり、データ定義のみで完結する範囲が大幅に広がります。

2.  **AI入力特徴量の動的構成 (Dynamic AI Input Feature Configuration)**
    *   **現状**: ニューラルネットワークへの入力テンソル（特徴量）の定義はC++コード内で固定（約200次元）されています。メタゲームの変化により特定のゾーン（例：墓地、超次元）の重要度が変わった場合、再コンパイルが必要です。
    *   **提案**: 特徴量の構成要素（自分の手札枚数、相手のマナ文明分布など）をJSON設定ファイルで定義し、実行時に動的にテンソルを構築する仕組みを導入します。
    *   **効果**: AIモデルのアーキテクチャ変更を柔軟に行えるようになり、自己進化エコシステムにおいて「AIがどの情報を重視すべきか」自体を探索・学習の対象にすることが可能になります。

3.  **完全な再現性を持つリプレイシステム (Fully Reproducible Replay System)**
    *   **現状**: エラー発生時のデバッグはテキストログに依存しており、複雑な相互作用やAIの挙動の再現が困難です。
    *   **提案**: 初期シード値、プレイヤーデッキ、およびアクションIDの列のみを記録した軽量なバイナリ形式のリプレイファイルを定義し、GUI上で任意の時点まで状態を復元・再生できるビューアを実装します。
    *   **効果**: 「数千試合に一度発生するバグ」の特定が容易になり、QAプロセスの効率が飛躍的に向上します。また、AIの「好プレー/珍プレー」の分析も容易になります。
