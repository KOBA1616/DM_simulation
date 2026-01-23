# 21. 改定ロードマップ: AI進化と統合 (AI Evolution & Integration)

## 概要 (Overview)
本ドキュメントは、「要件定義書 00」に基づき、エンジン刷新（Phase 6/7）完了後の新たな開発ロードマップを定義するものである。
エンジン基盤の安定化に伴い、凍結されていたAI開発フェーズ（Phase 2/3）を再開し、実戦的な強さの獲得を目指す。

---

## Phase 6 & 7: エンジン刷新 (Engine Overhaul) [Completed]
**ステータス: [Done/Archived]**

以下のエンジン刷新タスクは完了し、`main` ブランチに統合された。
*   **GameCommand Implementation**: アクションのコマンド化。
*   **GenericCardSystem**: `EffectResolver` からの移行。
*   **Hybrid Architecture**: 新旧ロジックの共存と段階的移行。

---

## Phase 2: 不完全情報推論 (Inference System)
**優先度: High [Status: WIP]**
**目的**: 相手の手札やシールドの内容を推論し、より人間らしい（または人間を超える）読み合いを実現する。

### Task 2.1: Inference Engine Integration
*   C++で実装済みの `DeckInference` クラスをPython側のエージェントに統合する。
*   `PimcGenerator` (Perfect Information Monte Carlo) を用いて、推論結果に基づく仮想世界のサンプリングを行う。

### Task 2.2: Meta-Data Learning
*   対戦相手のデッキタイプを推定するための学習データを収集・活用する。

---

## Phase 3: 自己進化エコシステム (Evolution Ecosystem)
**優先度: High [Status: WIP]**
**目的**: AIが自らデッキを改良し、環境（メタゲーム）を作り出すシステムを構築する。

### Task 3.1: Automated PBT Loop
*   Population Based Training (PBT) の完全自動化。
*   `python/training/verify_deck_evolution.py` を拡張し、継続的なリーグ戦と淘汰を行うサーバー/スクリプト群を整備する。

### Task 3.2: Dynamic Meta Definition
*   `data/meta_decks.json` を静的なファイルではなく、学習結果に基づいて動的に更新されるデータベースとして運用する。

---

## Phase 4: モデル高度化 (Advanced Model Architecture)
**優先度: Medium [Status: Todo]**
**目的**: 単純なMLP (Multi-Layer Perceptron) から、シーケンスを扱えるTransformerモデルへ移行する。

### Task 4.1: Transformer Model Implementation
*   Python (PyTorch) 側でのTransformerモデルの実装。
*   C++ `TensorConverter` からのトークン入力を受け取るパイプラインの構築。

### Task 4.2: Attention Visualization
*   AIが「どのカード」や「どのゾーン」に注目しているかを可視化するツールの開発（GUI統合）。

---

## Phase 5: Future AI Refinement
**優先度: Low**
*   **Lethal Solver v2**: 完全探索ベースの必勝読み。
*   **Reinforcement Learning Update**: PPOやMuZeroなど、より高度なアルゴリズムの適用検討。
