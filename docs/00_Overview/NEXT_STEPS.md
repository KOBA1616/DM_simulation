# 今後の実装方針 (Implementation Roadmap)

**最終更新**: 2026年1月22日  
**テスト通過率**: 98.3% (121 passed + 41 subtests / 123 total + 41 subtests)  
**目標**: 99%以上

---

## 📊 現状サマリー

### ✅ 完了フェーズ（アーカイブ化済み）
- **Phase 1-5**: Legacy Action削除完了 → Commands-onlyアーキテクチャ確立
- **Phase 4 (Week 2-3)**: Transformer基礎実装完了 → 1エポック学習通過確認
- **ゲームエンジン**: 基本機能実装完了
- **詳細**: [99_Completed_Tasks_Archive.md](./99_Completed_Tasks_Archive.md)

### 🚧 進行中
- **Phase 6**: 品質保証（残課題2件）
- **Phase 4**: Transformer本番統合準備中
- **Phase 3**: メタゲーム進化基盤実装中

---

## 🎯 優先順位別タスク (Priority Tasks)

**重要な方針変更（2026年1月9日）**: 以下の2つを最優先で開発

### 【NEW - 最最優先】GUI改善とカード効果テスト環境

**目標**: カード開発者体験の向上と効果検証の効率化  
**期間**: 2週間  
**優先度**: ⭐⭐⭐ Critical

#### 既存のテスト環境（利用可能）

✅ **1. Scenario Tools (シナリオツール)**
- **位置**: メインウィンドウ → ツールバー → "Scenario Mode"
- **機能**:
  - 特定のゲーム状態を保存/読み込み
  - 任意のカードを任意のゾーンに追加
  - ゾーンのクリア操作
  - シナリオファイル: `data/scenarios/*.json`
- **使用例**:
  ```
  1. メインウィンドウで "Scenario Mode" をON
  2. 左パネルの "Add Specific Card" でテストしたいカードを手札に追加
  3. ステップ実行で効果を確認
  4. "Save Current State" でシナリオ保存
  ```

✅ **2. Batch Simulation (バッチシミュレーション)**
- **位置**: ツールバー → "Batch Simulation"
- **機能**:
  - 保存したシナリオで100+ゲームを自動実行
  - 勝率・平均ターン数等の統計収集
  - Random/Heuristic/MLPモデルで評価
- **使用例**:
  ```
  1. Scenario Toolsでテストシナリオ作成
  2. Batch Simulationを開く
  3. シナリオ選択、Episodes=100、Evaluator=Heuristic
  4. Run → 統計結果を確認
  ```

✅ **3. Main Game Window (リアルタイムデバッグ)**
- **機能**:
  - ステップ実行で効果を一つずつ確認
  - MCTS AnalysisでAIの思考を可視化
  - Loop Recorderで無限ループ検出
  - Card Detail Panelで効果テキスト確認

#### 新規実装タスク

**Task 1: カード効果デバッグツール** (1週間)
1. **効果解決ステップビューア** (3日)
   - [ ] EffectResolverの実行ステップを表示
   - [ ] 各コマンドの入力/出力状態を詳細表示
   - [ ] ブレークポイント機能（特定コマンドで停止）

2. **変数ウォッチャー** (2日)
   - [ ] ゲーム変数のリアルタイム表示
   - [ ] 変数値の履歴トラッキング
   - [ ] 変数参照エラーの警告

3. **コマンドスタック詳細** (2日)
   - [ ] 現在のスタック状態を視覚化
   - [ ] CHOICEコマンドのオプション表示
   - [ ] ネスト構造の見やすい表現

**Task 2: カード効果検証システム** (1週間)
1. **静的解析ツール** (3日)
   - [ ] コマンド構造の妥当性チェック
   - [ ] 変数参照の整合性検証
   - [ ] 無限ループパターン検出
   - 成果物: `dm_toolkit/validation/card_validator.py`

2. **自動テスト生成** (2日)
   - [ ] カード定義からテストケース生成
   - [ ] 基本シナリオとエッジケースの自動生成
   - 成果物: `scripts/generate_card_tests.py`

3. **効果トレーサー** (2日)
   - [ ] 解決履歴のJSONエクスポート
   - [ ] 視覚的フローチャート生成
   - 成果物: `EffectTracer` + HTMLレポート

**成功基準**:
- ✅ 新規カードの効果を5分以内にテスト可能
- ✅ 90%以上のバグを実行前に検出
- ✅ デバッグ時間を50%削減

**関連ファイル**:
- [dm_toolkit/gui/simulation_dialog.py](../../dm_toolkit/gui/simulation_dialog.py)
- [dm_toolkit/gui/widgets/scenario_tools.py](../../dm_toolkit/gui/widgets/scenario_tools.py)
- [dm_toolkit/gui/app.py](../../dm_toolkit/gui/app.py)

---

### 【最優先 - Critical】Phase 4: Transformer本番統合（従来の計画）

**目標**: Transformerモデルを実戦投入し、AI性能を大幅向上  
**期間**: 2週間  
**前提**: Week 2-3基礎実装完了

#### Task 4.1: 学習最適化 (1週間)
**目標**: 学習効率とモデル性能の最大化

1. **ハイパーパラメータチューニング** (2日)
   - 学習率: 1e-4 → 1e-3 範囲で探索
   - ドロップアウト率: 0.1 → 0.2 範囲で調整
   - Warm-up epochs: 5-10 epochで検証

2. **バッチサイズ拡大** (1日)
   - 8 → 16 → 32 の段階的拡大
   - 各サイズでメモリ使用量と学習速度を記録
   - GPUメモリ使用量のプロファイリング

3. **学習率スケジューラ** (1日)
   - CosineAnnealingLR の導入
   - Warm-up + Decay スケジュールの実装
   - 学習曲線の可視化

4. **Early Stopping** (1日)
   - Validation lossベースのEarly Stopping
   - 忍耐期間 (patience): 10-20 epochs
   - ベストモデルの自動保存

**成果物**:
- ✅ 最適化されたハイパーパラメータセット
- ✅ 安定した学習曲線（loss低下確認）
- ✅ TensorBoardログと可視化

#### Task 4.2: 評価システム構築 (3日)
**目標**: 客観的な性能評価基盤の確立

1. **vs Random ベンチマーク** (1日)
   - 100ゲーム対戦の実装
   - 勝率、平均ターン数、平均ゲーム時間の記録
   - 目標: 85%以上の勝率

2. **vs MLP 比較** (1日)
   - 現行MLPモデルとの100ゲーム対戦
   - Eloレーティング的な相対評価
   - 目標: 55%以上の勝率

3. **パフォーマンス測定** (1日)
   - 推論速度のベンチマーク（目標: <10ms/action）
   - メモリ使用量の測定
   - CPU vs GPU 推論速度比較

**成果物**:
- ✅ 評価スクリプト (`evaluate_transformer.py`)
- ✅ ベンチマークレポート (JSON/Markdown)
- ✅ 性能指標ダッシュボード

#### Task 4.3: デプロイ準備 (3日)
**目標**: 本番環境での実行準備

1. **ONNXエクスポート** (1日)
   - PyTorchモデルのONNX変換
   - ONNX Runtimeでの推論検証
   - 精度確認（PyTorchとの出力比較）

2. **C++統合** (1日)
   - C++からONNXモデルをロード
   - `GameState` → Token Sequence → ONNX推論のパイプライン
   - PythonとC++の出力一致確認

3. **モデルバージョニング** (1日)
   - モデルファイル名にバージョン番号付与
   - メタデータ（ハイパーパラメータ、学習日時）の記録
   - モデルレジストリの構築

**成功基準**:
- ✅ vs MLP ≥ 55% 勝率
- ✅ 推論速度 < 10ms/action
- ✅ 24時間連続学習で安定動作
- ✅ ONNXモデルのC++統合完了

**関連ファイル**:
- [dm_toolkit/ai/agent/transformer_model.py](../../dm_toolkit/ai/agent/transformer_model.py)
- [python/training/train_transformer_phase4.py](../../python/training/train_transformer_phase4.py)
- [python/tests/ai/export_transformer_onnx.py](../../python/tests/ai/export_transformer_onnx.py)

### 【高優先 - High】Phase 3: メタゲーム進化

**目標**: 自己進化エコシステムの完成  
**期間**: 1週間  
**前提**: `PopulationManager`, `ParallelMatchExecutor` 基礎実装完了

#### Task 3.1: Evolution Ecosystem統合 (3日)
1. **自動PBTループ** (1日) [Done]
   - ✅ 世代交代ロジックの実装 (`EvolutionOperator`)
   - ✅ デッキ評価・選択アルゴリズム (`run_evolution_loop`)
   - ✅ 淘汰と突然変異のバランス調整

2. **リーグ戦システム** (1日)
   - Round-robin トーナメントの実装
   - Eloレーティング計算
   - ランキングシステム

3. **可視化とロギング** (1日)
   - 進化プロセスのグラフ化
   - デッキ多様性の追跡
   - 世代ごとのスナップショット保存

#### Task 3.2: 動的メタ定義 (2日)
1. **`meta_decks.json` 動的更新** (1日)
   - 進化結果からの自動抽出
   - 上位デッキのメタデータ記録
   - バージョン管理

2. **メタ分析** (1日)
   - デッキタイプの自動分類
   - 戦略パターンの抽出
   - 相性マトリクスの生成

**成功基準**:
- ✅ 100世代の自動進化実行
- ✅ デッキプールの多様性維持（Shannon entropy > 2.0）
- ✅ 上位デッキの自動抽出と記録

**関連ファイル**:
- [python/training/evolution_ecosystem.py](../../python/training/evolution_ecosystem.py)
- [python/training/verify_deck_evolution.py](../../python/training/verify_deck_evolution.py)

---

### 【中優先 - Medium】Phase 6: 品質保証完遂

**目標**: テスト通過率 99%以上達成  
**期間**: 3-5日

#### 残課題
1. **GUIスタブ完全修正** (1-2日)
   - 一部テストケースの修正
   - モックオブジェクトの完全性向上

2. **テキスト生成完全対応** (1-2日)
   - 残りのゾーン名変換対応
   - CHOICEコマンド内の表現統一

3. **Beam Search C++メモリ問題** (調査中)
   - 未初期化メモリの特定
   - 修正と検証

**関連ファイル**:
- [run_pytest_with_pyqt_stub.py](../../run_pytest_with_pyqt_stub.py)
- [dm_toolkit/gui/editor/text_generator.py](../../dm_toolkit/gui/editor/text_generator.py)
- [src/ai/](../../src/ai/)

---

### 【低優先 - Low】将来タスク

#### A. カードエディタ完成度向上
1. Logic Mask実装（入力矛盾防止）
2. バリデーションルール強化
3. UI/UX改善

**関連**: [docs/03_Card_Editor_Specs.md](../03_Card_Editor_Specs.md)

#### B. 不完全情報推論 (Phase 2)
1. `DeckInference` Python統合
2. PIMC (Perfect Information Monte Carlo) 実装
3. メタデータ学習

**関連**: [docs/00_Overview/20_Revised_Roadmap.md](./20_Revised_Roadmap.md)

#### C. エンジンメンテナンス
1. テストカバレッジ向上
2. `src/engine` リファクタリング
3. ドキュメント整備

---

## 📊 進捗追跡指標

### 主要KPI
- **テスト通過率**: 98.3% → 99%以上（目標）
- **Transformer性能**: vs MLP ≥ 55% 勝率
- **推論速度**: < 10ms/action
- **進化世代**: 100世代の安定実行

### マイルストーン
- ✅ Phase 1-5 完了 (2026年1月)
- ✅ Phase 4 Week 2-3 完了 (2026年1月)
- 🚧 Phase 4 本番統合 (開始予定)
- 🚧 Phase 3 進化エコシステム (開始予定)
- ⏳ Phase 6 完遂 (残課題2件)

---

## 📚 関連ドキュメント

- **マスター要件定義**: [00_Status_and_Requirements_Summary.md](./00_Status_and_Requirements_Summary.md)
- **完了フェーズ**: [99_Completed_Tasks_Archive.md](./99_Completed_Tasks_Archive.md)
- **Phase 4 詳細**: [07_Transformer_Implementation_Summary.md](./07_Transformer_Implementation_Summary.md)
- **Legacy Action**: [01_Legacy_Action_Removal_Roadmap.md](./01_Legacy_Action_Removal_Roadmap.md)
- **開発ポリシー**: [AGENTS.md](../../AGENTS.md)
**タスク**:
- [ ] DeckInferenceクラスの精度向上
- [ ] PimcGeneratorの最適化
- [ ] メタデータ学習の実装

#### 6. 高度なAIアルゴリズム (Phase 5)
**タスク**:
- [ ] Lethal Solver v2（完全探索ベース）
- [ ] PPO/MuZero等の適用検討
- [ ] Attention Visualization

---

## 📈 マイルストーン

### Milestone 1: テスト100%通過 (1週間以内)
- [x] Phase 6.1完了（テキスト生成）
- [x] Phase 6.2完了（GUIスタブ）
- [ ] Phase 6.3完了（Beam Search修正）
- [ ] テスト通過率 100%達成

### Milestone 2: AI進化システム稼働 (1ヶ月以内)
- [ ] Transformerモデル統合
- [ ] メタゲーム進化パイプライン稼働
- [ ] 継続的な自己対戦環境の構築

### Milestone 3: 本番リリース準備 (3ヶ月以内)
- [ ] 全機能のドキュメント完成
- [ ] パフォーマンスチューニング
- [ ] ユーザーガイドの作成

---

## 🔧 開発ワークフロー

### 推奨作業順序
1. **今週中**: Transformerモデル統合の仕上げ (Phase 4)
2. **来週**: メタゲーム進化パイプラインの構築 (Phase 3)
3. **継続的**: Beam Search修正とエディタ改善

### 並行作業の推奨
- Transformer開発 ∥ Evolution開発（Phase 3/4は並行可）

---

## 📚 関連ドキュメント

- [00_Status_and_Requirements_Summary.md](00_Status_and_Requirements_Summary.md) - 全体要件定義
- [01_Legacy_Action_Removal_Roadmap.md](01_Legacy_Action_Removal_Roadmap.md) - Phase 1-6詳細
- [20_Revised_Roadmap.md](20_Revised_Roadmap.md) - AI進化ロードマップ
- [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) - 開発プロセス

---

## ✅ 完了基準

各タスクは以下の基準で完了とみなす：
1. 実装完了とテスト通過
2. コードレビュー完了
3. ドキュメント更新
4. CI/CDパイプライン通過

---

**Note**: 本ドキュメントは実装状況に応じて随時更新されます。
