# GUIノード切り替え問題 修正レポート

**日付**: 2026年1月17日  
**問題**: グループ「ロジック」のアクションタイプが「カードを引く (DRAW_CARD)」になってしまう

---

## 🔍 問題の原因分析

### 根本原因

[unified_action_form.py](dm_toolkit/gui/editor/forms/unified_action_form.py) の `on_group_changed()` および `on_type_changed()` における以下の問題：

1. **グループ変更時のシグナル伝播問題**
   ```python
   # 修正前のコード
   def on_group_changed(self):
       # ...
       self.populate_combo(self.type_combo, types, ...)
       if types:
           self.type_combo.setCurrentIndex(0)  # ← シグナル発火！
   ```
   - `setCurrentIndex(0)` が `currentIndexChanged` シグナルを発火
   - `on_type_changed()` が意図せず呼ばれる
   - type_combo がまだ空の状態だと `currentData()` が None になる

2. **不正なフォールバック値**
   ```python
   # 修正前のコード
   def on_type_changed(self):
       t = self.type_combo.currentData()
       if t is None:
           t = "DRAW"  # ← これはグループ名！コマンドタイプではない！
   ```
   - "DRAW" はグループ名であって、コマンドタイプではない
   - 正しいコマンドタイプは "DRAW_CARD"、"NONE" など

3. **データロード時のタイミング問題**
   ```python
   # 修正前のコード
   self.set_combo_by_data(self.action_group_combo, grp)
   self.set_combo_by_data(self.type_combo, cmd_type)
   ```
   - グループを設定すると、`on_group_changed()` が呼ばれて type_combo が再構築される
   - その後 type_combo を設定しても、既に内容が変わっている可能性

---

## ✅ 実施した修正

### 修正1: `on_group_changed()` のシグナル制御

```python
def on_group_changed(self):
    grp = self.action_group_combo.currentData()
    types = COMMAND_GROUPS.get(grp, [])
    if not types:
        types = []

    prev = self.type_combo.currentData()
    
    # シグナルをブロックして、意図しない on_type_changed() の呼び出しを防ぐ
    self.type_combo.blockSignals(True)
    try:
        self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)

        if prev and prev in types:
            self.set_combo_by_data(self.type_combo, prev)
        else:
            if types:
                self.type_combo.setCurrentIndex(0)
    finally:
        self.type_combo.blockSignals(False)
    
    # 手動でUI再構築をトリガー
    self.on_type_changed()
```

**効果**:
- ✅ グループ変更中はシグナルをブロック
- ✅ 最後に明示的に `on_type_changed()` を1回だけ呼ぶ
- ✅ 意図しないシグナル発火を防止

### 修正2: `on_type_changed()` のフォールバック修正

```python
def on_type_changed(self):
    t = self.type_combo.currentData()
    if t is None:
        # 正しいフォールバック: コンボボックスの最初の項目またはNONE
        if self.type_combo.count() > 0:
            t = self.type_combo.itemData(0)
        if t is None:
            t = "NONE"  # 有効なコマンドタイプへフォールバック
    self.rebuild_dynamic_ui(t)
    self.update_data()
```

**効果**:
- ✅ グループ名ではなく、実際のコマンドタイプを使用
- ✅ 空のコンボボックスでも正しく処理

### 修正3: `_load_ui_from_data()` のタイミング改善

```python
# Block signals during combo updates
self.action_group_combo.blockSignals(True)
self.type_combo.blockSignals(True)

try:
    # グループを設定（これで type_combo が再構築される）
    self.set_combo_by_data(self.action_group_combo, grp)
    # type_combo を手動で再構築
    types = COMMAND_GROUPS.get(grp, [])
    self.populate_combo(self.type_combo, types, data_func=lambda x: x, display_func=tr)
    # 正しいタイプを設定
    self.set_combo_by_data(self.type_combo, cmd_type)
    
    # UI再構築
    self.rebuild_dynamic_ui(cmd_type)
finally:
    self.action_group_combo.blockSignals(False)
    self.type_combo.blockSignals(False)
```

**効果**:
- ✅ グループ変更後にtype_comboを確実に再構築
- ✅ データロード時のレースコンディションを回避

---

## 🔍 他のフォームの分析結果

### ✅ 問題なし: EffectEditForm

```python
def update_trigger_options(self, card_type):
    # ...
    self.trigger_combo.blockSignals(True)
    # ... populate ...
    self.trigger_combo.blockSignals(False)  # 正しくブロック解除
```

**評価**: シグナル制御が適切

### ✅ 問題なし: ModifierEditForm

```python
def update_visibility(self):
    mtype = self.type_combo.currentData()
    if mtype is None:
        mtype = "COST_MODIFIER"  # 有効なタイプへフォールバック
```

**評価**: フォールバック値が適切

### ✅ 問題なし: DynamicCommandForm

```python
def _load_ui_from_data(self, data, item):
    with self.suppress_signals():
        self.set_combo_by_data(self.type_combo, t)
        self.rebuild_form(t)  # 手動でトリガー
```

**評価**: `suppress_signals()` を正しく使用

### ✅ 問題なし: ReactionEditForm

```python
def set_combo_text(self, combo, text):
    idx = combo.findText(text)
    if idx >= 0:
        combo.setCurrentIndex(idx)
```

**評価**: BaseEditForm の `block_signals_all()` に依存

---

## 📊 修正の影響範囲

### 変更ファイル

| ファイル | 変更内容 | リスク |
|---------|---------|--------|
| [unified_action_form.py](dm_toolkit/gui/editor/forms/unified_action_form.py) | on_group_changed(), on_type_changed(), _load_ui_from_data() | 低 |

### 影響を受ける機能

| 機能 | 影響 | テスト推奨度 |
|------|------|------------|
| グループ切り替え (DRAW→LOGIC等) | ✅ 修正済み | ⭐⭐⭐ 必須 |
| コマンドタイプ切り替え | ✅ 改善 | ⭐⭐⭐ 必須 |
| データロード時のフォーム構築 | ✅ 安定化 | ⭐⭐ 推奨 |
| 新規アクション作成 | 影響なし | ⭐ 任意 |

---

## 🧪 検証項目

### 手動テストケース

1. **グループ切り替えテスト**
   - [ ] DRAW → LOGIC に変更
   - [ ] LOGICグループの最初のコマンド (QUERY) が選択されることを確認
   - [ ] "DRAW_CARD" が表示されないことを確認

2. **データロードテスト**
   - [ ] LOGIC/QUERY コマンドを保存
   - [ ] エディタを再起動または別ノードを選択後、再度選択
   - [ ] QUERYが正しくロードされることを確認

3. **エッジケーステスト**
   - [ ] 空のグループ (RESTRICTION) を選択
   - [ ] クラッシュせず、NONEまたは最初の項目が選択されることを確認

### 自動テストケース (推奨)

```python
def test_group_change_does_not_fallback_to_draw():
    form = UnifiedActionForm()
    
    # Set group to LOGIC
    form.set_combo_by_data(form.action_group_combo, "LOGIC")
    form.on_group_changed()
    
    # Verify type is not "DRAW_CARD"
    selected_type = form.type_combo.currentData()
    assert selected_type != "DRAW_CARD"
    assert selected_type in COMMAND_GROUPS["LOGIC"]
```

---

## 🎯 予防策と推奨事項

### 短期 (即座に実施)

1. ✅ **完了**: UnifiedActionForm の修正
2. 🔄 **推奨**: 手動テストの実施 (上記のテストケース)

### 中期 (1-2週間)

1. **自動テストの追加**
   - グループ/タイプ切り替えのユニットテスト
   - データロード/保存のラウンドトリップテスト

2. **コードレビューガイドラインの更新**
   - コンボボックス変更時は必ず `blockSignals()` を使用
   - フォールバック値は有効な値であることを確認

### 長期 (次回リファクタリング時)

1. **シグナル制御の統一化**
   - BaseEditForm に `with_signals_blocked(widget)` コンテキストマネージャを追加
   - 全フォームで統一的に使用

2. **フォールバック値の定数化**
   ```python
   DEFAULT_COMMAND_TYPE = "NONE"  # 集中管理
   ```

---

## 📝 関連する設計パターン

### 良い例: DynamicCommandForm

```python
with self.suppress_signals():
    self.set_combo_by_data(self.type_combo, t)
    self.rebuild_form(t)  # 明示的にトリガー
```

**学び**: コンテキストマネージャを使用して、シグナル制御のスコープを明確にする

### 改善された例: UnifiedActionForm (修正後)

```python
self.type_combo.blockSignals(True)
try:
    # ... 変更 ...
finally:
    self.type_combo.blockSignals(False)
# 手動でトリガー
self.on_type_changed()
```

**学び**: try-finally でシグナルブロックを確実に解除し、手動でイベントをトリガー

---

## ✅ 結論

**修正ステータス**: ✅ **完了**

- 根本原因: シグナル伝播と不正なフォールバック値
- 修正内容: シグナル制御の追加、フォールバック値の修正、データロードタイミングの改善
- 影響範囲: UnifiedActionForm のみ
- リスク: 低（ロジックの明確化、既存動作の改善のみ）

この修正により、グループ切り替え時に意図しないコマンドタイプへのフォールバックが発生しなくなります。
