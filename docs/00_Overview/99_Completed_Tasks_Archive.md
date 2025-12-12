# 完成したタスクのアーカイブ (Completed Tasks Archive)

このドキュメントは `00_Status_and_Requirements_Summary.md` から完了したタスクを移動して記録するためのアーカイブです。

## 完了済み (Completed)

### 2.4 実装上の不整合の修正 (Inconsistencies Resolved)

1.  **C++ コンパイル警告 (ConditionDef) の修正**
    *   `CostHandler`, `ShieldHandler`, `SearchHandler`, `DestroyHandler`, `UntapHandler` 等において、`ConditionDef` のブレース初期化リストを修正し、`missing initializer` 警告を解消しました。
    *   未使用のパラメータ (`unused parameter`) についても修正を行い、ビルドログをクリーンにしました。

2.  **Atomic Action テストの修正**
    *   `tests/python/test_new_actions.py` 内の `test_cast_spell_action` および `test_put_creature_action` を修正しました。
    *   `GenericCardSystem.resolve_effect_with_targets` を呼び出す際、明示的に `CardType` (SPELL/CREATURE) を設定した `card_db` を渡すことで、エンジンが正しい解決パス（呪文は墓地へ、クリーチャーはバトルゾーンへ）を選択するようにしました。

3.  **革命チェンジのデータ構造不整合の修正**
    *   `CardEditor` に「革命チェンジ」チェックボックスを追加し、Engineが期待するルートレベルの `revolution_change_condition` (FilterDef) を生成するように修正しました。
    *   チェックボックス操作に連動して、ロジックツリー内に `Trigger: ON_ATTACK_FROM_HAND` および `Action: REVOLUTION_CHANGE` を自動生成するロジックを実装しました。

4.  **文明指定のキー不整合の修正**
    *   `CardEditor` (Data Manager) が新規カード作成時にリスト形式の `"civilizations"` を使用するように統一しました。
    *   `CardEditForm` は既にリスト形式に対応していましたが、データの保存・読み込み時のレガシーサポート（`civilization` 文字列）は `JsonLoader` (Engine) 側で引き続き担保されます。

5.  **Card Editor UI の改善 (Polish)**
    *   **カードプレビュー**:
        *   ツインパクトカードのパワー表記を「カード全体の左下」に配置しました。
        *   マナコストの円形背景を文明色（多色の場合は等分割グラデーション）で描画するように修正しました。
        *   マナコストの文字色を「黒縁の白文字」（実装上は太字の白文字＋黒い円形枠線）に統一しました。
        *   カードの外枠（選択時の強調部分）を「すべての文明で黒の細線」に統一しました。
    *   **テキスト生成**: `CardTextGenerator` に `EX Life` (EXライフ) のキーワード対応を追加しました。
