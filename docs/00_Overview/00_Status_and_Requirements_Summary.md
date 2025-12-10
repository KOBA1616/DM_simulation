# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はShift-JISで記述することを前提とします（ツール制約によりファイルエンコードはUTF-8で保存されています）。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。
現在、Phase 0（基盤構築）および Phase 1（エディタ・エンジン拡張）を完了し、**Phase 2以降の「AI知能の進化と自己学習エコシステム」の構築**へフェーズを移行しています。

Python側のコードベースは `dm_toolkit` パッケージとして再構築され、モジュール性が向上しました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **フルスペック実装**: 基本ルールに加え、革命チェンジ、侵略、ハイパーエナジー、ジャストダイバー、ツインパクト、オレガ・オーラ（基礎）、封印（基礎）などの高度なメカニクスをサポート。
*   **アクションシステム**: `IActionHandler` による完全なモジュラー構造。新しい能力の追加はハンドラの実装のみで完結します。
*   **高速シミュレーション**: OpenMPによる並列化と最適化されたメモリ管理により、秒間数千〜数万試合の自己対戦が可能です。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.0**:
    *   **機能**: JSONデータの視覚的編集、ロジックツリー、変数リンク、テキスト自動生成。
    *   **多色・ツインパクト対応**: 複雑なカード構造を直感的に扱える専用UI（`CivilizationSelector`, `SpellSideWidget`）を実装済み。
    *   **検証済み**: 生成されたJSONデータはエンジンで即座に読み込み可能です。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **AlphaZero Pipeline**: データ収集(`DataCollector`) -> 学習(`train_simple.py`) -> 評価(`verify_performance.py`) のサイクルが稼働中。
*   **ResNet Model**: 現在の標準モデル。
*   **推論エンジン**:
    *   **Deck Classifier**: `meta_decks.json` を基に、公開情報から相手デッキタイプを確率的に特定。
    *   **Hand Estimator**: 相手手札にあるカードを確率的に推定するロジック（基礎実装済み）。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

今後は、AIが人間のような「読み」と「メタゲームへの適応」を行うための機能を実装します。

### 3.1 [Priority: High] Phase 2: 不完全情報の克服 (Imperfect Information / PIMC)

AIが「見えない領域（相手の手札、シールド、山札）」を確率的に推論し、最適な行動を決定するための機能群です。

1.  **相手手札推定器 (Opponent Hand Estimator)** [実装完了 / Verified]
    *   **ステータス**: `dm_toolkit/ai/inference` に実装済み。
    *   **機能**: `DeckClassifier` によるデッキ推定と、`HandEstimator` による手札確率計算（簡易ベイズ/頻度ベース）。
    *   **課題**: ゲーム状態に応じた動的なパラメータ（残り山札数、手札枚数）の連携精度向上。

2.  **PIMC (Perfect Information Monte Carlo) サーチ**
    *   **目的**: 不完全情報を「確定化（Determinization）」した複数の仮想世界で探索を行い、平均的に良い手を打てるようにする。
    *   **実装計画**:
        *   推論された確率分布に基づき、相手の手札・シールドをランダムに具体化（サンプリング）して `GameState` を生成する機能をC++エンジンに追加。
        *   並列探索(`ParallelRunner`)において、スレッドごとに異なる「仮想世界」を割り当てて探索し、ルートノードで結果を集約するロジックを実装する。

### 3.2 [Priority: Medium] Phase 3: 自筆進化エコシステム (Self-Evolving Ecosystem)

人間の介入なしに、AIが自らデッキを調整し、プレイスタイルを進化させる完全自動ループ（Population Based Training）を構築します。

1.  **自動学習サイクル (The Automation Loop)**
    *   **構成**: 以下の3プロセスを常時稼働させるスクリプト群を作成する。
        *   **Collector**: 複数のエージェント（Population）同士でリーグ戦を行い、学習データと勝率データを生成。
        *   **Trainer**: 最新のデータを用いてモデルを更新。
        *   **Gatekeeper**: 新モデルが旧モデルに対して有意に強い（勝率55%以上等）場合のみ `production` モデルを更新。

2.  **世代管理とストレージ戦略 (Generation Management)**
    *   **課題**: 無限に増える学習データとモデルによるディスク圧迫を防ぐ。
    *   **実装計画**:
        *   **階層化保存**:
            *   `checkpoints/production/`: 現在の最強モデル（常に1つ）。
            *   `checkpoints/population/`: 進化中の個体群（数十個、成績下位は自動削除）。
            *   `checkpoints/hall_of_fame/`: 過去の節目（Gen 10, 50, 100...）のモデルを永久保存し、強さのベンチマークとして利用。
        *   **データ・プルーニング**: 古い `.npz` 学習データ（例: 50世代前）を自動削除するガベージコレクション機能を実装。

3.  **デッキ進化 (Deck Evolution)**
    *   **目的**: 固定デッキでの対戦だけでなく、環境に合わせてデッキ内容を微修正する。
    *   **実装計画**:
        *   カードごとの「勝率貢献度（Win Contribution）」を算出し、貢献度の低いカードを候補プール（`candidate_pool`）のカードと入れ替える遺伝的アルゴリズムを実装。

### 3.3 [Priority: Future] Phase 4: アーキテクチャ刷新 (Architecture Update)

1.  **Transformer (Linear Attention) 導入**
    *   **目的**: 盤面のカード枚数が可変であるTCGの特性に合わせ、固定長入力のResNetから、可変長入力を扱えるAttention機構へ移行する。
    *   **計画**: `NetworkV2` として、PyTorchでのモデル定義と、C++側のテンソル変換ロジック（`TensorConverter`）の書き換えを行う。

### 3.4 [Priority: Low] Phase 5: エディタ機能の完成形 (Editor Polish)

AI開発と並行して、エディタの残存課題を解消します。

1.  **Logic Mask (ロジックマスク)**
    *   選択された「トリガー」に対して、意味的に無効な「効果」を選択できないようにGUI側でフィルタリングする機能。
2.  **Reaction Ability UI**
    *   ニンジャ・ストライクや革命0トリガーなどの「リアクション（手札誘発）」を編集するための専用フォームの実装。
3.  **Visual Effect Builder**
    *   ロジックツリーをノードグラフ形式で表示・編集できる高度なUI（長期目標）。
