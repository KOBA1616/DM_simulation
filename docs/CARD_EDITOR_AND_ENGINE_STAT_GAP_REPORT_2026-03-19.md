# カードテキスト統計トラッカー実装ギャップ報告書（カードエディタ/エンジン統合）

作成日: 2026-03-19  
更新日: 2026-03-22  
対象: `dm_toolkit/gui/editor`, `src/engine`, `data/cards.json`  
目的: DMカードテキスト解析で必要な統計情報を、カードエディタ定義からエンジン実行まで一貫管理するための不足点整理とTDD実装計画

---

## 残タスクチェックリスト

- [ ] 現時点で本計画範囲（P0-P3）の未完了タスクはありません

### 将来拡張候補（次サイクル管理）

- [ ] 再計算タイミング契約の監視強化（通常経路/MCTS経路の追加ケース）
- [ ] 統計キー追加時の監査項目の拡張（エディタ候補・評価器・データ監査の同時更新）
- [ ] CIワークフロー定義の整理（重複セクションの統合と監査テストの明示性向上）

---

## 1. 要約

`COMPARE_STAT` 評価と `COST_MODIFIER`/`PASSIVE` の実装軸が分かれていたため、
「統計値に応じて動的にコスト軽減する仕様」の運用契約が不十分でした。

P3の基盤整備は完了し、本計画範囲の残課題はありません。

---

## 2. 現状観測（根拠）

### 2.1 コスト軽減実装の分担

- Python試算: `dm_toolkit/payment.py`
- C++実運用: `ManaSystem::get_adjusted_cost` / `PaymentPlan::evaluate_cost` / `ContinuousEffectSystem`
- `COST_MODIFIER` の動的軽減（`STAT_SCALED`）は C++ 側で契約式を実装済み（エイリアス整合を含む）

### 2.2 静的条件の実装状況

- `ConditionValidator.VALID_STATIC_CONDITIONS` で `COMPARE_STAT` と `CARDS_MATCHING_FILTER` は許可済み
- 再計算タイミングと経路同一性の契約テストは導入済み（将来拡張は5章で管理）

### 2.3 統計キー接続の契約不足

- 統計キー追加時の整合は、監査テストとCI実行で継続監視する

### 2.4 二重適用リスク

- `validators_shared.detect_passive_static_conflicts` は `ERROR`/`WARNING` を出し分け
- `base_form` で `ERROR` 検出時に保存ブロック

### 2.5 存在確認系要件の汎用化余地

- `CARDS_MATCHING_FILTER` の件数比較で存在確認（`>= 1`）は共通化可能

---

## 3. 仕様定義（To-Be）

### 3.1 対応する軽減モード

1. `FIXED`
2. `STAT_SCALED`

推奨式:

`reduction = min(max_reduction, max(0, stat_value - min_stat + 1) * per_value)`

### 3.2 仕様フィールド

- `value_mode`: `FIXED` | `STAT_SCALED`
- `value`: `FIXED` 時必須
- `stat_key`: `STAT_SCALED` 時必須
- `per_value`: `STAT_SCALED` 時必須
- `min_stat`: 任意（既定1）
- `max_reduction`: 任意（未指定は上限なし）

### 3.3 合成順序契約

1. 基本コスト
2. `cost_reductions` の `PASSIVE`
3. `static_abilities` の `COST_MODIFIER`（`FIXED` / `STAT_SCALED`）
4. `ACTIVE_PAYMENT`
5. `min_mana_cost` フロア適用

### 3.4 エディタ要件

- `STATIC` 条件で `COMPARE_STAT` 選択可
- `stat_key` 候補は単一定義から供給
- `STAT_SCALED` 必須項目不足は保存前にブロック
- `FIXED` / `STAT_SCALED` の相互排他を検証

### 3.5 存在確認ベースの汎用条件

- `condition.type = CARDS_MATCHING_FILTER`
- `condition.op = ">="`
- `condition.value = 1`
- `condition.filter.owner = SELF`
- `condition.filter.zones = ["BATTLE_ZONE"]`
- `condition.filter.races = [<種族>]`

### 3.6 低スペックAI開発要件

- 1PR 1段階
- 1サイクル変更上限: 原則3ファイル
- 仕様追加は RED テスト先行
- 失敗時は feature flag で段階ロールバック

---

## 4. 低スペックAI向け開発プロトコル

### 4.1 作業単位テンプレ

- 入力: 1モード + 1計算経路
- 出力: 変更ファイル3つ以内
- 完了条件: RED追加 / GREEN化 / 再発防止コメント追加

### 4.2 実装コマンド最小セット

- `pytest tests/test_cost_modifier_stat_scaled.py -q`
- `pytest tests/test_cost_modifier_composition_order.py -q`
- `pytest tests/test_cost_reduction_spec_contract.py -q`

必要時:

- `pytest tests/test_payment_*.py -q`
- `pytest tests/ -q`

### 4.3 レビュー観点

- 合成順序とテストの一致
- `stat_key` のレジストリ整合
- `max_reduction` クランプ漏れ有無
- 0始まりプレイヤーID前提の維持

---

## 5. コスト軽減/常在効果の計算・処理タイミング再検討（将来拡張）

### 5.1 現状課題

- 統計更新直後の再計算契約が明示されていない
- MCTS one-shot 経路が通常経路と同一タイミングを常に踏む保証が弱い

### 5.2 再設計方針

契約:

1. 状態更新
2. 必要なら統計更新
3. `ContinuousEffectSystem::recalculate`
4. `generate_legal_commands` / `can_pay_cost`

### 5.3 計算責務分離

- 反映責務: `ContinuousEffectSystem::recalculate`
- 算出責務: `ManaSystem::get_adjusted_cost` / `CostPaymentSystem::can_pay_cost`

### 5.4 MCTS経路同一化

- `resolve_command_oneshot` 後に `recalculate`
- `fast_forward` 後のコスト判定前に再計算済みを保証

### 5.5 推奨シーケンス

通常経路:

1. コマンド適用
2. パイプライン
3. 統計更新
4. `recalculate`
5. 合法手生成

MCTS遷移:

1. one-shot適用
2. パイプライン
3. 統計更新
4. `recalculate`
5. `fast_forward`
6. 合法手生成

### 5.6 TDD固定契約

- `tests/test_cost_reduction_recalc_after_stat_update.py`
- `tests/test_mcts_cost_reduction_timing_parity.py`
- `tests/test_continuous_effect_recalc_before_generate_legal.py`

### 5.7 導入手順

1. テスト追加（RED）
2. MCTS one-shot 経路へ `recalculate` 追加
3. 通常経路/フェーズ経路の監査
4. コメント整理と文書更新
