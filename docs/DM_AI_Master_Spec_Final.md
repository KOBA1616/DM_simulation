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

### [11. 詳細実装ステップ (Detailed Implementation Steps)](./11_Detailed_Implementation_Steps.md)
- 開発の具体的な手順とチェックリスト。

### [12. PBT導入設計書 (PBT Design Spec)](./12_PBT_Design_Spec.md)
- **Population-Based Training**: Kaggle環境における自律進化型AI集団の構築設計。
- **Evolution Cycle**: 順次学習、リーグ戦、淘汰と変異のサイクル定義。
- **Niche Protection**: 多様性保護のためのロジック。

### [13. 非公開領域推論システム設計書 (POMDP Inference Spec)](./13_POMDP_Inference_Spec.md)
- **POMDP**: 部分観測マルコフ決定過程への対応。
- **Teacher-Student Distillation**: 知識の蒸留による「読み」の学習。
- **16-dim Stats**: 16次元スタッツによる確率推論。

### [14. メタゲーム・カリキュラム設計書 (Meta-Game Curriculum Spec)](./14_Meta_Game_Curriculum_Spec.md)
- **Dual Curriculum**: アグロとコントロールを交互に学習し、攻守のバランスを習得。
- **Meta-Game League**: 過去の自分や固定デッキとの対戦によるメタゲーム適応。
- **Adaptive Matchmaking**: 苦手な相手と優先的に戦うマッチメイキング。

### [15. 結果スタッツ設計書 (Result Stats Spec)](./15_Result_Stats_Spec.md)
- **Result Stats**: 対戦結果から算出される16次元のカード性能指標。
- **Autonomous Evolution**: 人間のタグ付けに頼らない自律的なデッキ構築と学習サイクル。
- **Kaggle Infrastructure**: クラウド資源を活用した24時間学習システム。

### [16. シナリオ・トレーニング設計書 (Scenario Training Spec)](./16_Scenario_Training_Spec.md)
- **Scenario Mode**: 特定の盤面から開始する「詰将棋」モード。
- **Combo Mastery**: 無限ループや即死コンボを短時間で習得させる特訓システム。
- **Loop Detection**: 同一局面の検知によるループ証明ロジック。

### [17. AI向け詳細実装指示書 (Detailed Implementation Instructions)](./17_Detailed_Implementation_Instructions_for_AI.md)
- **Step-by-Step Guide**: AIアシスタントが実装を行うための、コードレベルの具体的な指示書。
- **Phase 3-5 Coverage**: 結果スタッツ、POMDP、シナリオモード、PBTの実装詳細。

### [18. 実装計画の実現可能性検討書 (Feasibility Analysis)](./18_Feasibility_Analysis.md)
- **Gap Analysis**: 現状のコードベースと新要件の差分分析。
- **Risk Assessment**: 技術的リスクと対策の評価。
