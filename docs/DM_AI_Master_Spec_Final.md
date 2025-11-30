# Duel Masters AI Simulator - Ultimate Master Specification (v2.2)

本ドキュメントは、プロジェクトの全体仕様を管理するインデックスファイルです。詳細な仕様は各セクションのリンク先を参照してください。

## 目次 (Table of Contents)

### [1. プロジェクト概要 (Project Overview)](./01_Project_Overview.md)
- プロジェクトの目的、開発哲学、品質管理方針について記述しています。

### [2. システムアーキテクチャ (System Architecture)](./02_System_Architecture.md)
- 技術スタック、ディレクトリ構成、名前空間の設計について記述しています。

### [3. コア・データ仕様 (Core Data Specs)](./03_Core_Data_Specs.md)
- 定数、カードデータ構造、盤面状態の定義について記述しています。

### [4. ゲームルール詳細 (Detailed Game Rules)](./04_Game_Rules.md)
- ゲームの詳細なルールについて記述しています（現在はVer 2.0準拠）。

### [5. AI & 機械学習仕様 (AI/ML Specs)](./05_AI_ML_Specs.md)
- **Deep Sets / Set Transformer**: 順列不変なネットワーク構造。
- **LLM Text Embeddings**: カードテキストのベクトル化によるゼロショット適応。
- **AlphaZero Style MCTS**: C++による高速探索。
- **League Training**: 忘却防止のためのリーグ戦。
- **PBT**: ハイパーパラメータの自動最適化。
- **Auxiliary Tasks**: 補助タスクによる学習効率化。
- **PSRO**: メタゲームソルバーによるナッシュ均衡探索。

### [6. メタゲーム進化 (Meta-Game Evolution)](./06_Meta_Game_Evolution.md)
- **Quality-Diversity (MAP-Elites)**: 多様なデッキの発見。
- **Co-evolution**: デッキとプレイングの共進化。

### [7. GUI & 開発ツール (Frontend/DevOps)](./07_Frontend_DevOps.md)
- PyQt6によるGUI、可視化機能、開発補助ツールについて記述しています。

### [8. C++統合提案 (C++ Integration Proposal)](./08_Cpp_Integration.md)
- LibTorch統合、量子化、完全C++学習ループへのロードマップ。

### [9. 汎用カードジェネレーターとデータ駆動アーキテクチャ](./09_Card_Generator_Architecture.md)
- **ECS / Bytecode Interpreter**: データ駆動によるカード実装。
- **JSON Schema**: カード定義フォーマット。
- **Generator Tool**: GUI/AIによるカード生成ツール。

### [10. 開発状況と今後のロードマップ (Development Status & Roadmap)](./10_Development_Status_and_Roadmap.md)
- **Current Status**: Phase 2.5 (基盤構築完了・高度化準備)。
- **Future Roadmap**:
    - **Phase 3**: AIの高度化 (Deep Sets, LibTorch, PBT)。
    - **Phase 4**: コンテンツ拡充 (Card Generator)。
    - **Phase 5**: メタゲーム解明 (MAP-Elites, Co-evolution)。
