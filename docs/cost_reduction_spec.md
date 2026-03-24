# Cost Reduction（常在・能動）仕様

目的: 常在・能動のコスト軽減を、エディタ定義と Python/C++ 実行系で同一契約として扱う。

## 1. 適用対象

- `cost_reductions[]`（`PASSIVE`, `ACTIVE_PAYMENT`）
- `static_abilities[]` の `COST_MODIFIER`（`FIXED`, `STAT_SCALED`）
- コマンド入力（`reduction_id`, `payment_units`）

## 2. 合成順序契約

再発防止: 計算順序は **PASSIVE -> STATIC -> ACTIVE** を固定する。

1. 基本コスト
2. `cost_reductions` の `PASSIVE`
3. `static_abilities` の `COST_MODIFIER`（`FIXED` / `STAT_SCALED`）
4. `ACTIVE_PAYMENT`
5. `min_mana_cost` と 0 下限を適用

## 3. STAT_SCALED 契約

- `value_mode=STAT_SCALED` のとき `stat_key`, `per_value` を必須とする。
- 計算式は Python/C++ で共通とする。

`reduction = min(max_reduction, max(0, stat_value - min_stat + 1) * per_value)`

- `max_reduction` 未指定時は上限制限なし。
- 計算結果が 0 以下の場合は軽減を適用しない。

## 4. 段階ロールアウト / ロールバック

再発防止: 緊急ロールバック可能性を常に維持する。

- feature flag: `STAT_SCALED_ENABLED=1|0`
- 既定値: `1`（有効）
- `0` の場合、STAT_SCALED 軽減を Python/C++ の双方で無効化する。

## 5. 競合検出契約（エディタ）

- `detect_passive_static_conflicts` で `PASSIVE` と `COST_MODIFIER` の重畳を検査する。
- severity は以下の二段階:
  - `ERROR`: 無条件 PASSIVE と無条件 COST_MODIFIER の重畳
  - `WARNING`: 条件付き重畳で潜在競合の可能性がある場合
- `ERROR` を含む場合は保存をブロックする。

## 6. 実装参照

- Python: `dm_toolkit/payment.py`
- C++: `src/engine/systems/effects/continuous_effect_system.cpp`
- C++ 合成順: `src/engine/systems/mechanics/payment_plan.cpp`
- 競合検出: `dm_toolkit/gui/editor/validators_shared.py`
- 保存ブロック: `dm_toolkit/gui/editor/forms/base_form.py`

## 7. 契約テスト

- `tests/test_cost_modifier_stat_scaled.py`
- `tests/test_cpp_stat_scaled_formula_contract.py`
- `tests/test_stat_scaled_feature_flag.py`
- `tests/test_cpp_stat_scaled_feature_flag_contract.py`
- `tests/test_cpp_payment_plan_composition_contract.py`
- `tests/test_passive_static_conflict_validation.py`

## 8. 補足

この仕様は `docs/CARD_EDITOR_AND_ENGINE_STAT_GAP_REPORT_2026-03-19.md` の最終状態と同期して維持する。