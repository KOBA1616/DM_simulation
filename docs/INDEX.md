# DM Simulation ドキュメンテーション インデックス

## 📚 C++ Migration & AI Design Documents

作成日: 2026年2月7日

---

## 🎯 Phase 1 & 2 実装完了ドキュメント

### 実装レポート

1. **[PHASE1_IMPLEMENTATION_REPORT.md](reports/PHASE1_IMPLEMENTATION_REPORT.md)**
   - SimpleAI実装レポート
   - AI選択ロジック統一
   - 実装ファイル: simple_ai.hpp/cpp

2. **[PHASE2_IMPLEMENTATION_REPORT.md](reports/PHASE2_IMPLEMENTATION_REPORT.md)**
   - プレイヤーモード管理C++化
   - PlayerMode enum実装
   - GameState統合

3. **[PHASE1_AND_PHASE2_SUMMARY.md](reports/PHASE1_AND_PHASE2_SUMMARY.md)**
   - Phase 1 + 2 統合サマリー
   - 実装時間: 約2時間
   - コード削減: ~100行

### ビルド・テストスクリプト

4. **[test_phase1_simple_ai.py](../test_phase1_simple_ai.py)**
   - Phase 1 テストスクリプト
   - SimpleAI優先度テスト

5. **[test_phase2_player_modes.py](../test_phase2_player_modes.py)**
   - Phase 2 テストスクリプト
   - PlayerMode統合テスト

6. **[build_and_test_phase2.ps1](../build_and_test_phase2.ps1)**
   - ビルド・テスト自動化スクリプト

---

## 📋 アクション・優先度仕様

### アクションリファレンス

7. **[PLAYER_INTENT_REFERENCE.md](../PLAYER_INTENT_REFERENCE.md)** ⭐ **必読**
   - **全22種類のPlayerIntentアクション完全リファレンス**
   - 各アクションの詳細説明
   - パラメータ、生成条件、使用例
   - カテゴリ分類（ユーザー15 + 内部7）

8. **[action_quick_ref.md](action_quick_ref.md)**
   - アクション一覧表（クイックリファレンス）
   - 優先度別分類
   - フェーズ別生成アクション

### 優先度設計

9. **[PHASE_ACTION_PRIORITY_SPEC.md](../PHASE_ACTION_PRIORITY_SPEC.md)** ⭐ **必読**
   - **フェーズ別アクション優先度完全仕様**
   - 全7フェーズの詳細
   - 各フェーズの許可/マスクアクション
   - 効果解決の優先度設計

10. **[PRIORITY_QUICK_REFERENCE.md](../PRIORITY_QUICK_REFERENCE.md)**
    - 優先度クイックリファレンス
    - Level 1-5 分類
    - よくある間違い
    - デバッグチェックリスト

---

## 🚀 Phase 1.1 改善案（フェーズ対応AI）

### 設計ドキュメント

11. **[PHASE_AWARE_AI_DESIGN.md](../PHASE_AWARE_AI_DESIGN.md)**
    - フェーズ対応AI設計
    - 問題提起と解決策
    - 優先度マトリクス
    - 実装スケジュール

12. **[DESIGN_PHASE_AWARE_AI.hpp](../native_prototypes/DESIGN_PHASE_AWARE_AI.hpp)**
    - フェーズ対応SimpleAIヘッダー（サンプル）

13. **[DESIGN_PHASE_AWARE_AI.cpp](../native_prototypes/DESIGN_PHASE_AWARE_AI.cpp)**
    - フェーズ対応SimpleAI実装（サンプル）

---

## 📊 視覚的図解（Mermaid）

### アクション関連

14. **[action_classification.md](action_classification.md)**
    - アクション分類図
    - ユーザー/内部アクション
    - Atomicフロー図

15. **[action_generation_flow.md](action_generation_flow.md)**
    - アクション生成フロー図
    - 優先度レベル可視化

### 優先度関連

16. **[action_priority_flow.md](action_priority_flow.md)**
    - アクション選択フロー図
    - Level 1-4 優先順位

17. **[priority_gantt.md](priority_gantt.md)**
    - 優先度ガントチャート
    - 視覚的優先度表現

18. **[action_state_machine.md](action_state_machine.md)**
    - アクション状態遷移図
    - generate_legal_actions()フロー

---

## 🗺️ 全体計画

### マスタープラン

19. **[CPP_MIGRATION_PLAN.md](../CPP_MIGRATION_PLAN.md)**
    - Python→C++ 移行実装計画
    - Phase 1-5 ロードマップ
    - 実装優先順位マトリクス
    - **Phase 1 & 2: ✅ 完了**
    - Phase 3-5: 未着手

20. **[GAME_STARTUP_FLOW_ANALYSIS.md](../GAME_STARTUP_FLOW_ANALYSIS.md)**
    - ゲーム開始フロー分析
    - ファイル責務分割
    - 改善提案

---

## 🎓 学習リソース

### 初めての方へ

**推奨読書順序**:

1. 📖 [PLAYER_INTENT_REFERENCE.md](../PLAYER_INTENT_REFERENCE.md)
   - すべてのアクションを理解する

2. 📖 [action_quick_ref.md](action_quick_ref.md)
   - クイックリファレンスで確認

3. 📖 [PHASE_ACTION_PRIORITY_SPEC.md](../PHASE_ACTION_PRIORITY_SPEC.md)
   - フェーズ別の詳細仕様を学ぶ

4. 📖 [PRIORITY_QUICK_REFERENCE.md](../PRIORITY_QUICK_REFERENCE.md)
   - 優先度システムを理解

5. 📊 Mermaid図で視覚的に確認
   - [action_classification.md](action_classification.md)
   - [action_priority_flow.md](action_priority_flow.md)

### 実装者向け

**Phase 1.1実装ガイド**:

1. 📖 [PHASE_AWARE_AI_DESIGN.md](../PHASE_AWARE_AI_DESIGN.md)
   - 設計意図を理解

2. 💻 [DESIGN_PHASE_AWARE_AI.cpp](../native_prototypes/DESIGN_PHASE_AWARE_AI.cpp)
   - サンプル実装を確認

3. ✅ テストケース作成
   - `test_phase_aware_ai.py`

---

## 📈 実装ステータス

### ✅ 完了
- Phase 1: AI選択ロジック統一（SimpleAI）
- Phase 2: プレイヤーモード管理C++化（PlayerMode）

### 🚧 進行中
- Phase 1.1: フェーズ対応AI（設計完了、実装待ち）

### 📅 計画中
- Phase 3: イベント通知システム
- Phase 4: 自動進行スレッド化
- Phase 5: レガシーラッパー削除

---

## 🔍 クイック検索

### アクション名で検索

| 探したいこと | 参照ドキュメント |
|-------------|-----------------|
| アクション一覧 | [PLAYER_INTENT_REFERENCE.md](../PLAYER_INTENT_REFERENCE.md) |
| アクション詳細 | [PLAYER_INTENT_REFERENCE.md](../PLAYER_INTENT_REFERENCE.md) の各セクション |
| 優先度 | [PRIORITY_QUICK_REFERENCE.md](../PRIORITY_QUICK_REFERENCE.md) |
| フェーズ別アクション | [PHASE_ACTION_PRIORITY_SPEC.md](../PHASE_ACTION_PRIORITY_SPEC.md) |

### トピック別検索

| トピック | 参照ドキュメント |
|---------|----------------|
| SimpleAI実装 | [PHASE1_IMPLEMENTATION_REPORT.md](reports/PHASE1_IMPLEMENTATION_REPORT.md) |
| PlayerMode | [PHASE2_IMPLEMENTATION_REPORT.md](reports/PHASE2_IMPLEMENTATION_REPORT.md) |
| フェーズ対応AI | [PHASE_AWARE_AI_DESIGN.md](../PHASE_AWARE_AI_DESIGN.md) |
| 優先度設計 | [PHASE_ACTION_PRIORITY_SPEC.md](../PHASE_ACTION_PRIORITY_SPEC.md) |
| マスタープラン | [CPP_MIGRATION_PLAN.md](../CPP_MIGRATION_PLAN.md) |

---

## 📝 ドキュメント統計

- **総ドキュメント数**: 20
- **実装レポート**: 3
- **仕様書**: 5
- **設計書**: 3
- **視覚図**: 5
- **クイックリファレンス**: 2
- **計画書**: 2

---

## 🤝 貢献

### ドキュメント更新ガイドライン

1. Markdownフォーマットを維持
2. コードサンプルは実際のコードと一致させる
3. 視覚図（Mermaid）は単独ファイルで管理
4. このインデックスも更新する

---

**最終更新**: 2026年2月7日  
**バージョン**: 1.0  
**メンテナ**: AI Assistant
