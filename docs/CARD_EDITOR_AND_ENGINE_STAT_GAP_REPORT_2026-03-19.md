# カードテキスト統計トラッカー実装ギャップ報告書（カードエディタ/エンジン統合）

作成日: 2026-03-19  
対象: `dm_toolkit/gui/editor`, `src/engine`, `data/cards.json`  
目的: DMカードテキスト解析で必要な統計情報を、カードエディタ定義からエンジン実行まで一貫管理するための不足点整理とTDD実装計画

---

## 1. 要約

現状は「一部の統計条件（例: `OPPONENT_DRAW_COUNT`）」は実装済みですが、20,000枚規模を想定した統計網羅性・命名統一・置換効果との整合性が未達です。

特に問題なのは以下です。

- 条件種別/統計キーの定義源が分散し、エディタ表示・バリデーション・C++評価器の同期保証がない
- `TurnStats` が限定的で、報告書で必要な履歴統計（召喚回数、シールドブレイク実績、破壊実績など）が不足
- 置換効果適用時の「統計加算の抑止/振替」の仕様が明文化・テスト化されていない
- TDDが点在テスト中心で、統計トラッカー拡張に対する「契約テスト」が不足

本書では、上記を解消するための要件定義・段階実装・TODOを提示します。

---

## 2. 現状観測（根拠）

### 2.1 条件/統計キーの定義が限定的かつ分散

- `ConditionValidator` のトリガー条件は実質 `OPPONENT_DRAW_COUNT` 重点で、静的条件はさらに限定的  
  参照: `dm_toolkit/gui/editor/validators_shared.py`
- 条件フォーム定義 (`CONDITION_FORM_SCHEMA`) は少数条件のみ  
  参照: `dm_toolkit/gui/editor/schema_config.py`
- 条件UI候補 (`ConditionEditorWidget`) とスキーマ定義が二重管理  
  参照: `dm_toolkit/gui/editor/forms/parts/condition_widget.py`

### 2.2 `TurnStats` の網羅不足

`TurnStats` は現状、以下中心です。

- `played_without_mana`
- `cards_drawn_this_turn`
- `cards_discarded_this_turn`
- `creatures_played_this_turn`
- `spells_cast_this_turn`
- `mana_charged_by_player[2]`
- `player_draw_count[2]`

参照: `src/core/card_stats.hpp`

不足している代表例（本報告対象）:

- `summon_count_this_turn`（召喚のみ。踏み倒し/出すを分離）
- `mana_placed_this_turn`（手札チャージ + 効果加速の総量）
- `creatures_destroyed_this_turn`（置換で破壊不成立の場合は0加算）
- `shield_break_attempt_count_this_turn` と `shield_break_resolved_count_this_turn` の分離

### 2.3 `COMPARE_STAT` が参照可能なキーが少ない

`CompareStatEvaluator` は `MY/OPPONENT_*` のゾーン枚数系中心で、ターン履歴系・文明別統計・破壊/ブレイク系が未対応です。  
参照: `src/engine/systems/rules/condition_system.cpp`

### 2.4 `QUERY` 統計取得の範囲が限定的

`handle_get_stat` は `MANA_CIVILIZATION_COUNT`, `SHIELD_COUNT`, `HAND_COUNT`, `CARDS_DRAWN_THIS_TURN` などは対応済みですが、必要な履歴統計の多くは未対応です。  
参照: `src/engine/infrastructure/pipeline/pipeline_executor.cpp`

### 2.5 テキスト生成用統計マップが実装実態と乖離

`STAT_KEY_MAP` は `MANA_COUNT` 系を持つ一方で、`ConditionEditorWidget` の既定候補（`MY_MANA_COUNT` など）と完全一致していません。結果としてラベル化や自然文生成で未翻訳フォールバックが起こり得ます。  
参照: `dm_toolkit/gui/editor/text_resources.py`, `dm_toolkit/gui/editor/forms/parts/condition_widget.py`

### 2.6 置換効果連動の統計仕様が不足

置換効果コマンド (`REPLACE_CARD_MOVE`) 自体は存在しますが、

- ブレイク試行と実解決の統計分離
- 破壊置換時に `destroyed` 統計を加算しない規約

の統計契約が未定義です。  
参照: `data/cards.json`, `tests/test_effect_and_text_integrity.py`

---

## 3. ギャップ一覧（優先度付き）

| 優先度 | ギャップID | 不足点 | 影響 |
|---|---|---|---|
| P0 | G-STAT-001 | ターン履歴統計の不足（召喚/破壊/ブレイク分離） | 誘発条件誤判定、コスト軽減誤作動 |
| P0 | G-STAT-002 | 置換効果と統計更新順序の未契約化 | ルール裁定と挙動が乖離 |
| P0 | G-STAT-003 | 条件定義の分散（UI/validator/C++ evaluator） | 片側更新で破綻 |
| P1 | G-STAT-004 | `STAT_KEY_MAP` と評価器対応キーの不一致 | プレビュー誤表示、編集ミス誘発 |
| P1 | G-STAT-005 | QUERY統計の不足 | IF/選択ロジック実装が限定される |
| P1 | G-STAT-006 | 統計トラッカー契約テスト不足 | 回帰検知が遅い |
| P2 | G-STAT-007 | cards.json 側の未知コマンド/統計キー監査不足 | データ投入時の事故 |

---

## 4. 要件定義（To-Be）

### 4.1 統計トラッカーの正準仕様

`TurnStats` と `ZoneStats` を分離し、次のInternal IDを正準化する。

#### ターン履歴統計（毎ターンリセット）

- `STAT_DRAW_COUNT_T`
- `STAT_SUMMON_COUNT_T`
- `STAT_SPELL_CAST_T`
- `STAT_MANA_SET_T`
- `STAT_SHIELD_BREAK_ATTEMPT_T`
- `STAT_SHIELD_BREAK_RESOLVED_T`
- `STAT_CREATURE_DESTROYED_T`
- `FLAG_ATTACKED_THIS_T`

#### ゾーン/状態統計（イベント駆動更新）

- `STAT_GY_SIZE`
- `STAT_RACE_COUNT:<race_id>`
- `STAT_CIV_COUNT_MANA:<civ>`
- `STAT_MAX_POWER_B`
- `STAT_EVO_STACK:<instance_id>`
- `STAT_SEAL_COUNT:<instance_id>`

### 4.2 置換効果との整合契約

統計更新順序を以下で固定する。

1. 原イベント生成
2. 置換候補評価
3. 採用置換の確定
4. 実イベント適用
5. 実イベントベースで統計更新

再発防止コメントの必須追加箇所:

- 置換適用後のみ統計を加算する関数
- ブレイク試行/解決の分岐関数

### 4.3 エディタ要件

単一ソース化:

- 条件タイプ定義
- 統計キー定義
- キーごとの入力仕様（`value/op/stat_key/filter`）

必要機能:

- 未対応統計キーの入力を保存前にブロック
- `COMPARE_STAT` の候補を C++ 実装対応キーに限定
- 統計キー追加時に UI/validator/text generator の契約テストが同時必須

### 4.4 低スペックAI開発要件

- 1タスク1責務（1 PR = 1統計カテゴリ）
- 1サイクルの変更上限: 原則3ファイルまで
- すべての仕様追加は「先に契約テスト」を書いてから実装
- 失敗時は差分最小で戻す（大規模リファクタ禁止）

---

## 5. TDD実装計画

### Phase 0: 契約固定（RED）

- 追加: `tests/test_stat_tracker_contract.py`
- 追加: `tests/test_replacement_stat_semantics.py`
- 追加: `tests/test_condition_stat_key_registry.py`

REDで保証する項目:

- 未実装統計キーは明示失敗
- 置換時に `destroyed` / `shield_break_resolved` が誤加算されない
- エディタ候補キーと評価器キーが一致していなければ失敗

### Phase 1: C++最小実装（GREEN）

対象:

- `src/core/card_stats.hpp`
- `src/engine/systems/rules/condition_system.cpp`
- `src/engine/infrastructure/pipeline/pipeline_executor.cpp`

作業:

- `TurnStats` にP0統計を追加
- `COMPARE_STAT` / `QUERY` 参照を拡張
- 置換後イベント準拠の統計更新

### Phase 2: エディタ同期（GREEN）

対象:

- `dm_toolkit/gui/editor/schema_config.py`
- `dm_toolkit/gui/editor/forms/parts/condition_widget.py`
- `dm_toolkit/gui/editor/validators_shared.py`
- `dm_toolkit/gui/editor/text_resources.py`

作業:

- 統計キー定義を単一レジストリ化
- UI候補・バリデーション・ラベルの同一参照化

### Phase 3: リファクタ（REFACTOR）

- 旧キーエイリアスを `compat map` に隔離
- `ConditionEditorWidget` の `CONDITION_UI_CONFIG` 二重定義を削減
- 監査テストをCI必須化

### Phase 4: カードデータ移行

- `data/cards.json` の条件キー監査
- 必要時 `.migrated` 出力と差分レビュー
- 未知キーを fail-fast

---

## 6. TODOリスト（実行順）

-### P0

- [x] `TurnStats` に `STAT_SUMMON_COUNT_T` を追加（2026-03-19: `summon_count_this_turn` を追加、パイプライン/StatCommand に対応）
- [x] `TurnStats` に `STAT_MANA_SET_T` を追加（2026-03-19: `mana_set_this_turn` を追加、パイプライン/StatCommand に対応）
- [x] `TurnStats` に `STAT_SHIELD_BREAK_*` を追加（2026-03-19: `shield_break_attempt_count_this_turn` と `shield_break_resolved_count_this_turn` を追加）
- [x] `TurnStats` に `STAT_CREATURE_DESTROYED_T` を追加（2026-03-19: `creatures_destroyed_this_turn` を追加し、破壊遷移での加算を実装）
- [x] 置換効果適用後のみ統計更新する共通関数を導入（2026-03-19: `stat_update.hpp` を追加し `add_turn_destroyed_count` をエクスポート）
  - [x] 基本契約検証: 置換処理（`g_neo_activated`）が破壊カウント加算より前に実行されることを `tests/test_replacement_stat_semantics.py` で検証（2026-03-19）
- [x] `tests/test_replacement_stat_semantics.py` を追加し、破壊置換/ブレイク置換ケースを固定（2026-03-19）
- [x] `COMPARE_STAT` の対応キーをレジストリ駆動に変更（2026-03-19: `SUMMON_COUNT_THIS_TURN` / `DESTROY_COUNT_THIS_TURN` / `MANA_SET_THIS_TURN` を `CompareStatEvaluator` で評価するよう追加）


### P1

  - [x] エディタの条件/統計キー定義を単一ファイルへ統合（2026-03-19: `tests/test_condition_editor_single_source.py` を追加し `ConditionEditorWidget` が `CardTextResources` を参照することを契約テストで検証）
- [x] エディタの条件/統計キー定義を単一ファイルへ統合（2026-03-19: `COMPARE_STAT` 候補とエディタのクイック統計を `CardTextResources` に集約）
- [x] `STAT_KEY_MAP` を実装キーと完全同期（2026-03-19: `COMPARE_STAT` 候補キーを共通定義化し、`MY_*` ラベル欠落を解消）
  - [x] `QUERY(GET_STAT)` の対応統計を拡張（2026-03-19: `SUMMON_COUNT_THIS_TURN` を追加、エディタ/エンジン双方の契約テストを追加）
  - [x] `QUERY(GET_STAT)` の `MY_*` エイリアス対応を追加（2026-03-19: `pipeline_executor` に `MY_` プレフィックスのマッピングを追加し、`tests/test_query_get_stat_my_keys.py` を追加）
- [x] `DESTROY_COUNT_THIS_TURN` を追加（2026-03-19: エディタ/エンジン双方の契約テスト `tests/test_query_get_stat_destroyed.py` を追加）
- [x] `tests/test_condition_stat_key_registry.py` を追加（2026-03-19: キー同期/ラベル定義の契約テストを追加）

### P2

 - [x] `data/cards.json` に対する「未知統計キー監査」スクリプト追加（2026-03-19: `tools/stat_key_audit.py` を追加）
 - [x] cardsデータ移行手順書（差分検証手順付き）を docs 化（2026-03-19: `docs/cards_data_migration.md` を追加）
- [x] CIに統計契約テストジョブを追加（2026-03-19: GitHub Actionsワークフロー `stat-contract-tests` を追加）
  - [x] CIで `tools/stat_key_audit.py` を実行するステップを追加（2026-03-19）

追記（2026-03-19）: TurnStats への直接書き込み監査を実施しました。契約テスト `tests/test_turnstats_write_audit.py` を追加し、現在のコードベースでは未承認の直接書き込みは検出されませんでした。引き続き新規実装時にはこのテストを通すことを必須としてください。

---

## 7. 低スペックAI向け開発プロトコル

### 7.1 作業単位テンプレ

- 入力: 「統計1種類 + 条件1種類」だけを対象
- 出力: 変更ファイル3つ以内
- 完了条件:
  - REDテスト追加
  - GREEN化
  - 再発防止コメント追加

### 7.2 実装コマンド最小セット

- `pytest tests/test_stat_tracker_contract.py -q`
- `pytest tests/test_replacement_stat_semantics.py -q`
- `pytest tests/test_condition_stat_key_registry.py -q`

必要時のみ:

- `pytest tests/ -q`

### 7.3 レビュー観点（必須）

- 統計名がエディタ/C++/テストで一致しているか
- 置換前イベントで統計を加算していないか
- 0始まりプレイヤーID前提を崩していないか

---

## 8. Definition of Done

- P0 TODOが完了
- 置換効果統計テストが全てGREEN
- 統計キーの単一ソース化が完了
- cards.json未知キー監査がCIで実行される
- 追加/修正コードに再発防止コメントがある

---

## 9. 初手実装の推奨バッチ

最初の1バッチは次を推奨。

1. `STAT_SHIELD_BREAK_ATTEMPT_T` / `STAT_SHIELD_BREAK_RESOLVED_T` を実装
2. 置換有無で両者が分かれるテストを追加
3. エディタで `COMPARE_STAT` 候補に2キー追加

このバッチは影響が明確で、低スペックAIでも追跡しやすい。
