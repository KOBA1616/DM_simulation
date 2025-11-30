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

### Phase 3: 基盤の高速化と柔軟化 (Foundation for Scale)
**目的**: 数千種類のカードと数百万回の対戦に耐えうる「速度」と「拡張性」を確保する。
1.  **C++ Inference (LibTorch)**:
    - Python側の推論をC++ (LibTorch) に移行する。
    - **Note**: 量子化（Int8）は実装難易度が高いため**採用せず**、**Float32** での実装とする。
    - **Risk Management**: ビルド環境構築の難易度が高いため、トラブル時は**実装を後回し（凍結）**とし、Python推論での開発を継続する。
2.  **Generic Card System (Interpreter) & Refactoring**:
    - JSON定義を読み込んで動作する `GenericCard` クラスとエンジンを実装。
    - **Special Effects Handler**: JSONで表現しきれない特殊効果（Extra Win, Extra Turn等）は、**別ファイル（C++）で管理する専用のハンドラ**として実装し、JSONからフックできるようにする。
    - **Refactoring**: 現在の登録カード種類が少ないため、**既存カードも全てJSON定義へ移行（リファクタリング）** し、コードベースを統一する。これにより保守性を最大化する。
3.  **Card Generator GUI Tool**:
    - 開発者がGUI操作でカード効果を定義し、JSONを自動生成するツールを開発する。
    - **Features**:
        - **Visual Editor**: トリガー、条件、効果をノードやリストで直感的に編集。
        - **LLM Assistant**: **Gemini API** を利用し、自然言語からJSONを自動生成する機能。

### Phase 4: 知能の深化とコンテンツ拡充 (Intelligence & Expansion)
**目的**: AIに「カードの意味」を理解させ、カードプールを一気に拡大する。
1.  **LLM Text Embeddings (NLP × RL)**:
    - カードテキストをベクトル化して入力するパイプラインを構築。
    - Phase 3で構築したカードシステムと連携し、新カード追加時の「ゼロショット適応」を実現する。
2.  **Auxiliary Tasks (補助タスク)**:
    - 勝敗以外の予測タスク（次状態予測など）を追加し、スパースな報酬問題を解決する。
3.  **カード量産 (Mass Production)**:
    - ジェネレーターを用いて主要カード（100〜200種）を実装し、環境を多様化させる。

### Phase 5: メタゲームと進化 (Evolution & Meta-Game)
**目的**: 人間の介入なしに、最強デッキとプレイングを自動発見する。
1.  **League Training & PSRO**:
    - 計算リソースを **リーグ学習（League Training）** に集中させる。
    - 過去の自分や多様なエージェントとの対戦をスケジューリングし、メタゲームの均衡点（ナッシュ均衡）を目指す。
    - **Note**: PBT（Population Based Training）は計算コストが過大なため、本フェーズでは優先度を下げ（Optional）、リーグ学習による強さの向上を優先する。
2.  **Quality-Diversity (MAP-Elites) & Co-evolution**:
    - デッキ構築とプレイングを共進化させ、未知のコンボや「変な勝ち方」をするデッキを発掘する。

## 10.3 詳細実装計画 (Detailed Plan)
低級モデルやコンテキスト制限のある環境での開発を支援するため、タスクを極小単位に分解した詳細計画書を作成した。
今後の実装は原則として以下のドキュメントに従って進めること。

- **[11. 詳細実装計画書 (Detailed Implementation Steps)](./11_Detailed_Implementation_Steps.md)**
