# カードテキスト統計トラッカー実装ギャップ報告書（カードエディタ/エンジン統合）

作成日: 2026-03-19  
更新日: 2026-03-22  
対象: `dm_toolkit/gui/editor`, `src/engine`, `data/cards.json`  
目的: DMカードテキスト解析で必要な統計情報を、カードエディタ定義からエンジン実行まで一貫管理するための不足点整理とTDD実装計画

---

## 1. 要約

現状は「統計条件の評価（`COMPARE_STAT`）」と「コスト軽減（`COST_MODIFIER` / `PASSIVE`）」が別軸で実装されており、
**常在コスト軽減をゲーム統計値に応じて動的に変化させる仕様**が未定義です。

特に以下が未完了です。

- ネイティブ（C++）経路で `STAT_SCALED` を含む常在コスト軽減の契約を安定して満たすこと
- `cost_reductions(PASSIVE)` と `static_abilities(COST_MODIFIER)` の優先/合成仕様を実装と文書で固定すること
- エディタ・Python評価・C++評価で同一ルールを継続的に担保する契約テストを拡充すること

本書では、上記を解消するための要件定義・段階実装・TODOを提示します。

---

## 2. 現状観測（根拠）

### 2.1 コスト軽減実装の分断

- Python側の試算ロジックは `dm_toolkit/payment.py` にあり、`PASSIVE` と `ACTIVE_PAYMENT` を評価
- C++側の実運用は `ManaSystem::get_adjusted_cost` / `PaymentPlan::evaluate_cost` / `ContinuousEffectSystem` で分担
- `COST_MODIFIER` は `active_modifiers` として反映されるが、軽減量は固定値（`mod_def.value`）

### 2.2 静的条件の実装状況

- `ConditionValidator.VALID_STATIC_CONDITIONS` では `COMPARE_STAT` と `CARDS_MATCHING_FILTER` が許可済み
- 一方で、ネイティブ経路を含む統合評価の安定化（再計算タイミングと契約テスト）は継続課題

### 2.3 統計キー定義と評価の接続不足

- `COMPARE_STAT` の評価器は C++ 側で実装されているが、
  常在コスト軽減へ接続する設計（閾値型/比例型）が定義されていない
- 統計キー追加時の影響面（エディタ候補、バリデーション、エンジン計算）を同時に守る契約が不足

### 2.4 二重適用リスク

- `validators_shared.detect_passive_static_conflicts` は警告を出すが、
  仕様としての優先順・合成規則が明文化されていない

### 2.5 存在確認系の表現が専用化されやすい

- 要件「自分のバトルゾーンに[種族]があればコストを[3]軽減」は実運用上頻出だが、
  現状はカードごとの専用条件として実装されやすく、再利用性が低い
- 既存の `CARDS_MATCHING_FILTER` 評価器は「件数比較」を持っているため、
  `count >= 1` で存在確認を共通化できる余地がある

---

## 3. ギャップ一覧（優先度付き）

| 優先度 | ギャップID | 不足点 | 影響 |
|---|---|---|---|
| P0 | G-COST-STAT-003 | 合成順序（PASSIVE / STATIC / ACTIVE）の契約未定義 | 二重適用・過小適用の回帰 |
| P1 | G-COST-STAT-004 | エディタ候補キーとエンジン評価キーの同期契約不足 | 保存は通るが実行時不整合 |
| P1 | G-COST-STAT-005 | 統計連動軽減の契約テスト不足 | 将来変更での回帰検知遅延 |
| P2 | G-COST-STAT-006 | cards.json 側の新フィールド監査ルール不足 | データ投入時の品質低下 |

---

## 4. 要件定義（To-Be）

### 4.1 対応する軽減モード

`COST_MODIFIER` の軽減を以下2モードで扱う。

1. `FIXED`（閾値型）
   - 既存互換。条件が真なら `value` を適用
   - 例: 「このターン自分が2回以上攻撃していればコスト-1」

2. `STAT_SCALED`（比例型）
   - 統計値に応じて軽減量を算出
   - 例: 「このターンの召喚回数ぶん軽減（最大3）」

推奨式:

`reduction = min(max_reduction, max(0, stat_value - min_stat + 1) * per_value)`

### 4.2 仕様フィールド（後方互換あり）

`static_abilities` の `COST_MODIFIER` で以下を許可:

- `value_mode`: `FIXED` | `STAT_SCALED`（未指定は `FIXED`）
- `value`: 固定軽減量（`FIXED` 時必須）
- `stat_key`: 参照統計キー（`STAT_SCALED` 時必須）
- `per_value`: 単位軽減量（`STAT_SCALED` 時必須）
- `min_stat`: 発動下限統計（任意、既定1）
- `max_reduction`: 軽減上限（任意、未指定なら上限なし）

### 4.3 合成順序契約

コスト計算の合成順を以下で固定する。

1. 基本コスト
2. `cost_reductions` の `PASSIVE`
3. `static_abilities` の `COST_MODIFIER`（`FIXED` / `STAT_SCALED`）
4. `ACTIVE_PAYMENT`
5. `min_mana_cost` フロア適用

再発防止コメントの必須追加箇所:

- 合成順を実装する関数（`ManaSystem` / `PaymentPlan`）
- `STAT_SCALED` 算出関数（負値防止、上限クランプ）

### 4.4 エディタ要件

- `STATIC` 条件で `COMPARE_STAT` を選択可能にする
- `stat_key` 候補は `CardTextResources` の単一定義から供給
- `STAT_SCALED` 時の必須フィールド不足を保存前にブロック
- `value_mode=FIXED` と `STAT_SCALED` の相互排他バリデーションを実施

### 4.5 存在確認ベースの汎用条件要件

以下を専用コマンドではなく、汎用条件として定義する。

- 要件例: 「自分のバトルゾーンに[種族]があればコストを[3]軽減する」

標準表現:

- `condition.type = CARDS_MATCHING_FILTER`
- `condition.op = ">="`
- `condition.value = 1`
- `condition.filter.owner = SELF`
- `condition.filter.zones = ["BATTLE_ZONE"]`
- `condition.filter.races = [<種族>]`

この形式により、以下を同一ルールで表現可能とする。

- 存在する: `>= 1`
- 存在しない: `== 0`
- N体以上: `>= N`
- ちょうどN体: `== N`

### 4.6 低スペックAI開発要件

- 1PR 1段階（Validatorのみ / Engineのみ / Editorのみを分割）
- 1サイクル変更上限: 原則3ファイルまで
- すべての仕様追加は RED テスト先行
- 失敗時は機能旗（feature flag）で段階ロールバック可能にする

---

## 5. TDD実装計画

### Phase 0: 契約固定（RED）

追加テスト:

- `tests/test_static_cost_modifier_compare_stat_allowed.py`
- `tests/test_cost_modifier_stat_scaled.py`  (追加: Python側RED→GREEN済、`evaluate_cost` の `STAT_SCALED` 算出とクランプを検証)
- `tests/test_cost_modifier_composition_order.py`
- `tests/test_editor_cost_modifier_stat_scaled_fields.py`
- `tests/test_static_cost_modifier_cards_matching_filter.py`

REDで保証する項目:

- `STATIC` 条件で `COMPARE_STAT` が拒否される現状を失敗で可視化
- `STAT_SCALED` フィールド未設定時に保存が失敗
- 合成順序が変わると失敗
- `CARDS_MATCHING_FILTER` による存在確認（`>= 1`）で軽減が有効化される

### Phase 1: エディタ/バリデータ最小解放（GREEN）

対象:

- `dm_toolkit/gui/editor/validators_shared.py`
- `dm_toolkit/gui/editor/schema_config.py`
- `dm_toolkit/gui/editor/text_resources.py`

作業:

- `VALID_STATIC_CONDITIONS` に `COMPARE_STAT` を追加
- `VALID_STATIC_CONDITIONS` に `CARDS_MATCHING_FILTER` を追加
- `COST_MODIFIER` へ `value_mode` / `stat_key` / `per_value` / `min_stat` / `max_reduction` の編集項目を追加
- 必須/型バリデーション追加
- `CARDS_MATCHING_FILTER` 用の `filter + op + value` 入力を静的能力で許可

進捗:

- `tests/test_editor_cost_modifier_stat_scaled_fields.py` を追加（`DM_DISABLE_NATIVE=1` をテスト内で設定して純Pythonフォールバックを使用）。
- `ModifierValidator` における `STAT_SCALED` 必須フィールド検証がRED→GREENになりました（`stat_key`, `per_value` の必須検査、および `per_value>0` 検査を含む）。

### Phase 2: Python試算系の拡張（GREEN）

対象:

- `dm_toolkit/payment.py`
- `tests/test_payment_*` 系

作業:

- `STAT_SCALED` の軽減算出ヘルパを追加
- `FIXED` と `STAT_SCALED` を統一計算
- 合成順（PASSIVE → STATIC → ACTIVE）を契約化

進捗:

- `evaluate_cost` の合成順序を明示的に実装（PASSIVE -> STATIC -> ACTIVE）。
- `tests/test_cost_modifier_composition_order.py` を追加し、PASSIVE と STATIC の適用順序を検証してGREEN化。

 - `apply_passive_reductions` を修正して、明確に「explicit PASSIVE のみ」を適用するように変更（静的能力の変換は `evaluate_cost` 側で処理）。これによりツールキットの予測経路がエンジン契約に一致します。
 - `tests/test_apply_passive_reductions_order.py` を追加して RED→GREEN の TDD サイクルを実施（テスト PASS）。

CI:

- GitHub Actions ワークフロー `.github/workflows/ci.yml` を追加しました。プッシュ/PR 時に `pytest` を Linux/Windows で実行し、テスト実行環境では `DM_DISABLE_NATIVE=1` を設定して純 Python フォールバックで安定実行する構成です。CI により今回追加した契約テストが継続的に回るようになります。

データ監査:

- `tests/test_cards_stat_scaled_audit.py` を追加しました。これは `data/cards.json` をスキャンし、`static_abilities` 内の `COST_MODIFIER` で `value_mode=STAT_SCALED` の場合に `stat_key` と `per_value>0` が存在することを検証します。CI 組み込みにより、欠落があるカードデータはプルリクで失敗するようになります。
追加テスト（統計キー同期）:

- `tests/test_stat_key_editor_engine_sync.py` を追加。`CardTextResources` の `COMPARE_STAT_EDITOR_KEYS` と `STAT_KEY_MAP` のキーを `ModifierValidator` の `STAT_SCALED` 設定で検証し、現在のバリデータがそれらのキーを受け入れることを確認（GREEN）。
- 注: 現状 `ModifierValidator` は未知の `stat_key` も許容する（レジストリ整合性チェックは別レイヤで運用）。

### Phase 3: C++本実装拡張（GREEN）

対象:

- `src/core/card_json_types.hpp`
- `src/engine/systems/effects/continuous_effect_system.cpp`
- `src/engine/systems/mechanics/mana_system.cpp`
- `src/engine/systems/mechanics/payment_plan.cpp`

作業:

- JSON定義へ新フィールドを追加（後方互換デフォルトあり）
- `STAT_SCALED` 軽減を `ManaSystem` 側で都度評価
- 合成順とフロア処理を統一

### Phase 4: リファクタ（REFACTOR）

- Python/C++で重複する計算式をドキュメント化し一致保証
- 旧 `value` 単独定義カードを自動移行可能な互換レイヤを追加
 - `stat_key` 不明時 fail-fast + 明確なエラーメッセージ（Validatorで検証を追加、テストで検証済み）

### Phase 5: データ移行・監査

- `data/cards.json` の `COST_MODIFIER` エントリ監査
- `STAT_SCALED` で必須フィールド不足カードを検出
- CIに監査ステップを追加

---

## 6. TODOリスト（実行順）

### 未完了タスク（優先順）

### P0

- [ ] ネイティブ（C++）経路で `STAT_SCALED` の end-to-end 契約テストを安定してGREEN化する
- [ ] 合成順（PASSIVE → STATIC → ACTIVE）を C++ 実装側でも明示し、回帰テストで固定する
- [ ] `tests/test_cpp_stat_scaled_integration.py` をネイティブ有効環境で常時検証できる状態にする

### P1

- [x] `stat_key` の候補定義（エディタ）と評価定義（Python/C++）の同期契約テストを追加しました（tests/test_stat_key_editor_engine_sync.py）。
- [ ] `onnxruntime` バージョン差分に依存しない CI ポリシー（固定 or xfail 方針）を明文化する
 - [x] `onnxruntime` バージョン差分に依存しない CI ポリシー（固定 or xfail 方針）を明文化しました（docs/ONNXRUNTIME_CI_POLICY.md）。

### P2

- [ ] cards監査を CI で必須ゲート化し、`STAT_SCALED` の必須項目欠落をブロックする
 - [x] cards監査を CI で必須ゲート化し、`STAT_SCALED` の必須項目欠落をブロックする（`tests/test_cards_stat_scaled_audit.py` を追加）



---

## 7. 低スペックAI向け開発プロトコル

### 7.1 作業単位テンプレ

- 入力: 「1モード（FIXED/STAT_SCALED）+ 1計算経路（Python/C++）」
- 出力: 変更ファイル3つ以内
- 完了条件:
  - REDテスト追加
  - GREEN化
  - 再発防止コメント追加

### 7.2 実装コマンド最小セット

- `pytest tests/test_static_cost_modifier_compare_stat_allowed.py -q`
- `pytest tests/test_cost_modifier_stat_scaled.py -q`
- `pytest tests/test_cost_modifier_composition_order.py -q`

必要時のみ:

- `pytest tests/test_payment_*.py -q`
- `pytest tests/ -q`

### 7.3 レビュー観点（必須）

- 合成順序がコードとテストで一致しているか
- `stat_key` がエディタ/エンジンで同じレジストリ基準か
- `max_reduction` の上限クランプ漏れがないか
- 0始まりプレイヤーID前提を崩していないか

---

## 8. Definition of Done

- 未完了TODO（本書 6 章）が完了
- `STAT_SCALED` 契約テストが全てGREEN
- エディタとエンジンで `stat_key` 参照定義が一致
- cards.json監査がCIで実行される
- 追加/修正コードに再発防止コメントがある

---

## 9. コスト軽減/常在効果の計算・処理タイミング再検討

 

本章は「いつ計算するか」を再設計し、MCTS検討時・通常対戦時で同じ結論が得られるようにするための方針を定義する。

### 9.1 現状課題（タイミング観点）

- 常在効果再計算（`ContinuousEffectSystem::recalculate`）はイベント駆動で呼ばれるが、
  統計更新の直後に必ず走る契約にはなっていない
- MCTSの one-shot 遷移では、通常経路と同じ再計算タイミングを必ずしも踏まない可能性がある
- その結果、
  - 行動候補生成時
  - 実行時
  - シミュレーション遷移時
  で有効な軽減値がズレるリスクがある

### 9.2 再設計方針（単一タイミングモデル）

「コストに影響する状態が変わったら、次の可否判定の前に再計算済みである」ことを契約にする。

状態変化イベント（最小集合）:

- ゾーン移動（入場/離脱/破壊/進化元変化）
- タップ状態変化（条件に使う場合）
- ターン統計更新（攻撃回数、召喚回数、ドロー等）
- フェーズ遷移（DURING_* 条件）

契約:

1. 状態更新
2. 必要なら統計更新
3. `ContinuousEffectSystem::recalculate`
4. 次の `generate_legal_commands` / `can_pay_cost` を実行

### 9.3 計算責務の分離

- 反映責務: `ContinuousEffectSystem::recalculate`
  - 盤面/条件から `active_modifiers` を再構築
- 計算責務: `ManaSystem::get_adjusted_cost` / `CostPaymentSystem::can_pay_cost`
  - 現時点の `active_modifiers` を使って最終コストを算出

再発防止:

- `get_adjusted_cost` 内で「不足分を推測して補正」するのではなく、
  前段で再計算済みであることを前提にする（責務混在を避ける）

### 9.4 MCTS 経路の同一化

MCTSでも通常ゲーム経路と同じ順序を保証する。

- `resolve_command_oneshot` 後に `ContinuousEffectSystem::recalculate` を実行
- `fast_forward` でフェーズ進行した場合、コスト判定前に再計算済みであることを保証

これにより「探索木上の合法手」と「実対戦の合法手」の乖離を抑制する。

### 9.5 推奨処理シーケンス（擬似仕様）

#### 通常経路

1. コマンド適用
2. パイプライン実行
3. 統計更新
4. `recalculate`
5. 合法手生成

#### MCTS遷移

1. one-shot コマンド適用
2. パイプライン実行
3. 統計更新
4. `recalculate`
5. `fast_forward`
6. 合法手生成

### 9.6 TDDで固定する契約

- `tests/test_cost_reduction_recalc_after_stat_update.py`
  - 統計更新後に軽減値が反映されること
  - 追加: Python側の `evaluate_cost` を対象にしたテストを追加し、統計値変更時の軽減再計算を検証（tests/test_cost_reduction_recalc_after_stat_update.py）。
- `tests/test_mcts_cost_reduction_timing_parity.py`
  - 同一状態で MCTS遷移と通常遷移のコスト判定が一致すること
- `tests/test_continuous_effect_recalc_before_generate_legal.py`
  - 合法手生成直前に再計算済みであること

### 9.7 導入手順（低リスク）

1. まずテスト追加（RED）
2. MCTS one-shot 経路に `recalculate` を追加（最小変更）
3. 通常経路/フェーズ経路の再計算ポイントを監査
4. 最後に責務混在のコメント整理とドキュメント更新

この順序なら挙動差分を限定しつつ、コスト軽減と常在効果のタイミングを安定化できる。



