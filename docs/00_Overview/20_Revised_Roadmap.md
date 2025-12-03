# 20. 改定ロードマップ: 堅牢性と拡張性を重視した開発計画 (Revised Roadmap)

## 20.1 概要 (Overview)
本ドキュメントは、2025-12-07時点での再定義された開発ロードマップである。
以前のロードマップ（Phase 2.5/3/4）を再構成し、以下の4つの重要方針に基づきタスクを順序付ける。

1.  **堅牢性 (Robustness) & ロールバックの容易性**: 開発を小さな単位で区切り、バグ発生時の手戻りを最小化する。
2.  **基本ロジック優先 (Basic Logic First)**: 高度な学習（PBT/League）の前に、エンジンと単純な学習ループの確実な動作を保証する。
3.  **拡張性 (Extensibility)**: 特に「カード実装」において、エンジンコードを修正せずにGUIから新カードを追加できる仕組みを早期に確立する。
4.  **小規模全体試験 (Small-scale Overall Test)**: 「対戦→データ収集→学習→性能向上」のサイクルが回ることを、限定されたシナリオで実証する（MVPサイクル）。

---

## Phase 1: 基盤と堅牢性の確保 (Foundation & Robustness)
**目的**: 現在確認されている技術的課題（Import Error等）を解消し、エンジンの動作を保証するテスト基盤を確立する。

### Task 1.1: 環境とビルドの安定化
- [x] **Python Import Issue**: Windows/MinGW環境等で報告されているDLLロードエラー等の問題を調査・解決し、`import dm_ai_module` が確実に成功する状態にする。
- [x] **Build Pipeline**: `cmake .. && make` の手順で確実に最新バイナリが生成され、Pythonから参照できる状態を維持する。

### Task 1.2: エンジン・ユニットテストの拡充
- [ ] **Core Logic Tests**: `GameState` (マナチャージ、シールドブレイク等) の基本動作を検証する `pytest` を追加する。
- [ ] **CardStats Verification**: 実装済みの `CardStats` が正しくデータを集計しているか、テストケースで検証する。

---

## Phase 2: 拡張性の確立 - GUIカード実装システム (Extensibility: Card Generator)
**目的**: ユーザーがGUI操作でカードを追加し、それを即座にゲームで使用できる「データ駆動型開発」の環境を構築する。これは「汎用性・拡張性」の要件を満たすための最優先事項である。

### Task 2.1: JSONベースのカード定義の実装
- [ ] **GenericCardSystem Integration**: `src/core/card_json_types.hpp` で定義された構造体を、C++エンジンが読み込み、実際のゲームロジックとして実行できることを確認する（まずは手書きJSONで検証）。

### Task 2.2: Python GUI カードエディタの開発
- [ ] **Advanced Card Editor**: 既存の `card_editor.py` (CSVベース) を廃止または大幅改良し、JSON形式 (`Trigger`, `Condition`, `Actions`) を視覚的に編集できるGUIツールを作成する。
- [ ] **Preview & Validation**: エディタ上でカードのJSON構造が正しいか検証する機能。

### Task 2.3: エディタとエンジンの連携確認
- [ ] **Integration Test**: GUIで作ったカード（例: 「登場時1ドロー」のクリーチャー）を保存し、再コンパイルなしでエンジンが読み込み、ゲーム内で効果が発動することを確認する。

---

## Phase 3: 小規模全体試験 - MVPサイクル (MVP Cycle)
**目的**: シンプルなAIエージェントが「学習によって強くなる」現象を、限定された環境（シナリオモード）で確認する。

### Task 3.1: AIアーキテクチャの実装 (AI Architecture)
- [x] **TensorConverter Update**: 入力テンソルにおいて、相手の非公開情報（手札・山札）をマスクする機能と、全公開（Full Info）で出力する機能を実装する。
- [x] **Heuristic Agent**: 学習データの元となるルールベースのエージェント（マナカーブ通りに出す、殴れるなら殴る等）を作成する。
- [x] **MCTS (Policy + Mask)**: 不確定情報を平均化して扱う「Masked MCTS」を実装する。

### Task 3.2: データ収集と学習ループの構築
- [x] **Data Collection**: Heuristic Agent同士の対戦を行い、`Masked State -> Action` (Policy) および `Full State -> Result` (Value) のペアを収集する。
- [x] **Supervised Training**: 収集したデータを用いて、AlphaZeroネットワークを教師あり学習させる。

### Task 3.3: 性能向上の検証 (Verification)
- [ ] **Impact Analysis**: 学習後のモデルが、学習前（ランダムまたはHeuristic）よりも勝率が向上していることを数値で証明する。

---

## Phase 4: コンテンツ拡充と高度化 (Expansion & Advanced AI)
**目的**: Phase 3で確立したフレームワーク上で、量産と高度な学習を行う。
**注記**: エンジン機能の大幅な拡張（Declaration Zone, Stackなど）はこのフェーズで設計・実装される。

### Task 4.1: エンジン機能拡張 (Engine Expansion)
- [ ] **Generic Tap/Untap**: 汎用的なタップ・アンタップ効果の実装。
- [ ] **Shield Burn**: シールド焼却能力の実装。
- [ ] **Underlying Cards**: 進化元や封印など、カードの下にカードを重ねる構造の実装。

### Task 4.2: カード量産
- [ ] **Mass Production**: GUIエディタを用いて、主要カード（100種〜）をJSON化する。

### Task 4.3: 高度な学習機能の導入
- [ ] **Self-Play Loop**: Heuristicデータからの脱却。自己対戦による強化学習ループの構築。
- [ ] **PBT**: ハイパーパラメータ自動探索。

---

## Phase 5: エンジンリファクタリング (Engine Refactor)
**目的**: 非常に複雑な処理（ハイパーエナジー、G・ゼロ、無月の門など）に対応するため、エンジンのコアプロセスを再設計する。

### Task 5.1: 宣言ステップとスタックの導入
- [ ] **Declaration Zone**: カード使用宣言を行う仮想ゾーンの導入。
- [ ] **Cost Calculation System**: 宣言されたカードに対し、コスト軽減や代替コスト（タップ要求）を解決するフェーズの実装。
- [ ] **Stack System**: 効果解決待ちの状態を管理するスタック領域の実装。

## 開発の進め方とロールバック方針
1.  **Step-by-Step**: 各Taskは独立したPR（Pull Request）またはコミットとして管理し、「動作する状態」を常に維持する。
2.  **Verification First**: コードを書く前に「どう検証するか（テストコード/検証スクリプト）」を定義する。
3.  **Config over Code**: ロジックの変更が必要な場合、ハードコード修正ではなく、設定ファイル（JSON）で制御可能にできないか検討する。
