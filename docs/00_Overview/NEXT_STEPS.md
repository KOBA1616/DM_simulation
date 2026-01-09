# 今後の課題と優先順位 (Next Steps & Priorities)

**最終更新**: 2026年1月22日
**テスト通過率**: 98.3% (121 passed + 41 subtests / 123 total + 41 subtests)

---

## 📊 現状サマリー

### ✅ 完了済み（Phase 1-6）
- **Legacy Action削除**: フェーズ1-5完了（入口統一、データ移行、GUI撤去、互換撤去、デッドコード削除）
- **Command Pipeline**: 統一的なコマンドシステムの確立
- **ゲームエンジン**: 基本機能の実装完了
- **AI学習基盤**: AlphaZero学習ループの実装
- **テキスト生成**: 自然言語化の強化 [Phase 6.1]
- **GUIスタブ**: ヘッドレステスト環境の整備 [Phase 6.2]

### 🚧 進行中
- **Phase 4**: Transformerモデル統合
- **Phase 3**: メタゲーム進化システム

---

## 🎯 優先度別タスク

### 【優先度: 高 - Critical】Phase 4 Transformer

#### 1. Transformerモデルの学習統合 (Phase 4.4)
**影響**: AI性能の大幅向上  
**作業量**: 大（1週間）

**タスク**:
- [ ] `train_transformer_phase4.py`の完成度向上
  - データローダーの最適化
  - GPU学習の安定化
- [ ] TensorConverterとの連携強化
  - パディング/マスキングのC++側処理との整合性確認
- [ ] パフォーマンステストとベンチマーク
  - 推論速度 < 10ms の達成確認

**ファイル**: 
- `dm_toolkit/ai/agent/transformer_model.py`
- `python/training/train_transformer_phase4.py`

---

#### 2. メタゲーム進化パイプライン (Phase 3)
**影響**: 自己対戦による継続的改善  
**作業量**: 中（1週間）

**タスク**:
- [ ] `evolution_ecosystem.py`の本番統合
- [ ] 自動PBTループの実装
- [ ] `data/meta_decks.json`の動的更新メカニズム
- [ ] リーグ戦システムの構築
- [ ] 結果の可視化とロギング

**ファイル**: 
- `python/training/evolution_ecosystem.py`
- `python/training/verify_deck_evolution.py`（参考実装）

---

### 【優先度: 中 - Enhancement】エンジン改善

#### 3. Beam Searchの修正
**影響**: AI探索性能  
**作業量**: 中（調査込み）

**タスク**:
- [ ] C++評価器の未初期化メモリ問題の特定
- [ ] メモリ初期化の修正
- [ ] テストのスキップ解除: `test_beam_search.py::test_beam_search_logic`
- [ ] パフォーマンス測定とベンチマーク

**ファイル**: 
- `src/ai/` (C++側)
- `python/tests/ai/test_beam_search.py`

---

#### 4. カードエディタの完成度向上
**影響**: 開発体験、データ品質  
**作業量**: 中（1週間）

**タスク**:
- [ ] Logic Maskの実装（入力矛盾の防止）
- [ ] バリデーションルールの強化
- [ ] UIフィードバックの改善
- [ ] テンプレートの拡充

**ファイル**: 
- `dm_toolkit/gui/editor/`
- [docs/03_Card_Editor_Specs.md](../03_Card_Editor_Specs.md)

---

### 【優先度: 低 - Future】長期改善

#### 5. 不完全情報推論の強化 (Phase 2)
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
