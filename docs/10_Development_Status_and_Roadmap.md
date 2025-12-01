# 10. 開発状況と今後のロードマップ (Development Status & Roadmap)

## 10.1 現在の開発段階 (Current Status: Phase 2.5)
**「基盤構築完了・高度化仕様策定フェーズ」**

現在、シミュレーターの核となるC++エンジンとPython GUIの連携は完了し、基本的な対戦が可能な状態にある。
直近の検討により、最強AI構築とカード量産のための**高度な機能群（AI/ML Specs, Card Generator）の要件定義が完了**したが、これらは**未実装**である。

### 実装済み機能 (Implemented)
- **Core Engine (C++20)**:
    - ビットボード、メモリプール、ゼロコピー転送による高速エンジン。
    - 基本的なゲームルール、フェーズ進行、ゾーン管理。
- **GUI (PyQt6)**:
    - 盤面表示、操作、MCTS可視化、ドック形式のレイアウト最適化。
    - `KeyboardInterrupt` 対応などのユーザビリティ向上。
- **Basic AI**:
    - 基本的なMCTS + MLP（AlphaZeroベース）の動作確認。

### 仕様策定済み・未実装機能 (Specified / Not Yet Implemented)
以下の機能は要件定義書（v2.2）に追加されたが、コード実装はこれから行う。
- **Advanced AI Architecture**:
    - **LLM Text Embeddings**: カードテキストのベクトル化入力。
- **Advanced Training Methods**:
    - **Auxiliary Tasks**: 勝敗以外の予測タスク。
    - **PBT**: ハイパーパラメータ自動最適化。
    - **PSRO**: メタゲームソルバー。
    - **League Training**: リーグ学習システム。
- **Meta-Game Evolution**:
    - **QD (MAP-Elites)**: 多様性探索。
    - **Co-evolution**: デッキとプレイングの共進化。
- **Development Tools**:
    - **Generic Card Generator**: データ駆動型カード実装システム（ECS/インタプリタ）。

## 10.2 今後の開発計画 (Future Roadmap)

### Phase 3: AIコアの進化 (AI Core Evolution)
**目的**: 「結果スタッツ」と「非公開領域推論」を実装し、AIの基礎知能を飛躍的に向上させる。
1.  **Result Stats System (Spec 15)**:
    - C++エンジンに `CardStats` 構造体を実装し、16次元のスタッツを収集・ベクトル化する。
    - 人間のタグ付けを廃止し、データ駆動でのカード評価基盤を確立する。
2.  **POMDP & Distillation (Spec 13)**:
    - 相手の手札・シールドを推論するための Teacher-Student 蒸留システムを構築する。
    - C++エンジンに「自分自身の山札確率」を計算する `Self-Inference` ロジックを実装する。

### Phase 4: 高度な学習手法 (Advanced Training)
**目的**: 複雑なコンボやメタゲームの駆け引きを習得させる。
1.  **Scenario Training (Spec 16)**:
    - 特定盤面から開始する「詰将棋モード」を実装し、無限ループやリーサル手順を特訓する。
2.  **Meta-Game Curriculum (Spec 14)**:
    - アグロ/コントロールを交互に学習する「Dual Curriculum」を導入。
    - 苦手な相手と優先的に戦う「Adaptive League」を構築する。

### Phase 5: 自律進化エコシステム (Autonomous Ecosystem)
**目的**: 人間の介入なしに最強デッキを発見し続けるシステムを完成させる。
1.  **PBT & Kaggle Integration (Spec 12)**:
    - Kaggle Notebooks 上で動作する PBT (Population Based Training) 環境を構築。
    - 24時間稼働によるハイパーパラメータ探索とデッキ進化を実現する。

### Parallel Track: コンテンツ拡充 (Content Expansion)
**目的**: カードプールを拡大し、環境の多様性を確保する。
1.  **Generic Card Generator (Spec 9)**:
    - GUI操作とLLM補助によるカード量産ツールを開発する。
    - 既存カードをJSON定義へ完全移行する。

## 10.3 詳細実装計画 (Detailed Plan)
低級モデルやコンテキスト制限のある環境での開発を支援するため、タスクを極小単位に分解した詳細計画書を作成した。
今後の実装は原則として以下のドキュメントに従って進めること。

- **[11. 詳細実装計画書 (Detailed Implementation Steps)](./11_Detailed_Implementation_Steps.md)**
