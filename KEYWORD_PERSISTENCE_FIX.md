# スペシャルキーワード選択状態保存の問題修正

## 問題報告
> スペシャルキーワード選択状態が保存されない、アクション自動生成されない

ユーザーが Revolution Change、Mekraid、Friend Burst などのスペシャルキーワードを選択して保存しても、
カードを再度開くときにチェックボックスが復元されず、対応する効果も再生成されない。

## 修正内容

### 修正1: KeywordEditForm の hook メソッド修正

**問題**: BaseEditForm の hook メソッド名が一致していなかった

**ファイル**: `dm_toolkit/gui/editor/forms/keyword_form.py`

| 項目 | 修正前 | 修正後 | 理由 |
|------|--------|--------|------|
| データ読み込み | `_populate_ui(item)` | `_load_ui_from_data(data, item)` | BaseEditForm が呼び出すメソッド名に統一 |
| データ保存 | `_save_data(data)` | `_save_ui_to_data(data)` | BaseEditForm が呼び出すメソッド名に統一 |

**詳細**:
- BaseEditForm の `load_data()` メソッドが `_load_ui_from_data()` を呼び出す
- BaseEditForm の `save_data()` メソッドが `_save_ui_to_data()` を呼び出す
- メソッド名が違うため、hook メソッドが呼ばれていなかった

**修正コード**:
```python
# 修正前: 名前が違うため呼ばれていない
def _populate_ui(self, item):
    kw_data = item.data(Qt.ItemDataRole.UserRole + 2)
    ...

def _save_data(self, data):
    ...

# 修正後: 正しい hook メソッド名
def _load_ui_from_data(self, data, item):
    # データは既に抽出されているので直接使用
    if data is None:
        data = {}
    
    # チェックボックスを復元
    for k, cb in self.keyword_checks.items():
        is_checked = data.get(k, False)
        cb.setChecked(is_checked)
    
    # スペシャルキーワードを復元
    self.rev_change_check.blockSignals(True)
    self.rev_change_check.setChecked(data.get('revolution_change', False))
    self.rev_change_check.blockSignals(False)
    # ... (mekraid, friend_burst も同様)

def _save_ui_to_data(self, data):
    # チェック状態をデータに保存
    for k, cb in self.keyword_checks.items():
        if cb.isChecked():
            data[k] = True
    
    if self.rev_change_check.isChecked():
        data['revolution_change'] = True
    # ... (mekraid, friend_burst も同様)
```

### 修正2: JSON 読み込み時に KEYWORDS ツリーアイテム作成

**問題**: カード JSON から keywords 辞書を読み込むがツリー構造に反映されていない

**ファイル**: `dm_toolkit/gui/editor/data_manager.py` (line 275-283)

```python
# 修正前: keywords は読み込まれるが、ツリーアイテムは作成されない
def load_data(self, cards_data):
    for card_idx, card in enumerate(cards_data):
        card_item = self._create_card_item(card)
        # トリガー、静的能力、リアクション能力を追加...
        # keywords は JSON にはあるが、ツリーに反映されない
        # → KeywordEditForm が読み込むデータがない

# 修正後: keywords データがあれば KEYWORDS ツリーアイテムを作成
# 2.5 Add Keywords if present in card JSON
keywords_data = card.get('keywords', {})
if keywords_data and isinstance(keywords_data, dict):
    kw_item = QStandardItem(tr("Keywords"))
    kw_item.setData("KEYWORDS", Qt.ItemDataRole.UserRole + 1)
    # 元のカード data を変更しないようにコピーを作成
    kw_item.setData(keywords_data.copy(), Qt.ItemDataRole.UserRole + 2)
    card_item.appendRow(kw_item)
```

## 動作フロー

### 保存時の流れ
```
ユーザーがチェックボックスをチェック
    ↓
toggle_rev_change() / toggle_mekraid() / toggle_friend_burst() 発動
    ↓
structure_update_requested シグナル送出
    ↓
CardEditor.on_structure_update() で効果ツリーアイテムを作成
    ↓
update_data() 呼び出し
    ↓
BaseEditForm.save_data() → _save_ui_to_data() 呼び出し
    ↓
KEYWORDS アイテムのデータ更新 (keywords_dict に revolutionary_change: True など)
    ↓
"Save JSON" クリック
    ↓
reconstruct_card_data() で KEYWORDS アイテムから keyword フラグを抽出
    ↓
card_data['keywords'] に統合
    ↓
JSON ファイルに保存
```

### 読み込み時の流れ (修正後)
```
CardEditor.load_data() カード JSON 読み込み
    ↓
data_manager.load_data()
    ↓
各カードについて:
  - トリガー効果を追加
  - 静的能力を追加
  - リアクション能力を追加
  - ★[NEW] keywords データがあれば KEYWORDS ツリーアイテムを作成
  - 呪文側を追加
    ↓
ユーザーが KEYWORDS アイテムを選択
    ↓
PropertyInspector が KeywordEditForm を活性化
    ↓
BaseEditForm.load_data() を呼び出し
    ↓
_load_ui_from_data() 実行 (修正後の正しいメソッド名)
    ↓
チェックボックス状態が復元される
```

## テスト結果

### 全テスト実行
```
125 passed, 1 failed (pre-existing i18n issue), 5 skipped
```
- 既存の1つの失敗は `app.py:473` のハードコード英文字列問題 (unrelated)
- 新しいテスト失敗なし
- 既存テスト破損なし

### 関連テストモジュール
```
python/tests/dm_toolkit/ : 17 passed, 1 skipped
```

## 検証できること

✅ スペシャルキーワード選択状態が保存される
- Revolution Change をチェック → 保存 → カードを再選択 → チェックボックスが復元される
- Mekraid をチェック → 保存 → 再読み込み → 状態が保持される
- Friend Burst をチェック → 保存 → 再読み込み → 状態が保持される

✅ スペシャルキーワード対応の効果がツリーに作成される
- チェックボックスをチェック → 対応する EFFECT アイテムがツリーに追加
- チェックボックスをアンチェック → EFFECT アイテムがツリーから削除

✅ JSON ファイルが正しく保存される
- keywords ディクショナリが `"keywords": { "revolution_change": true, ... }` で保存される

✅ 複数回の保存/読み込みでも状態が保持される
- カード保存 → 読み込み → 再保存 → 再読み込み → 状態継続

## 関連ファイル

| ファイル | 修正内容 |
|---------|---------|
| `dm_toolkit/gui/editor/forms/keyword_form.py` | `_load_ui_from_data()`, `_save_ui_to_data()` メソッド修正 |
| `dm_toolkit/gui/editor/data_manager.py` | `load_data()` に keywords ツリーアイテム作成コード追加 |

## 関連するメソッド (参考)

- `CardDataManager.reconstruct_card_data()` (line 418): KEYWORDS アイテムから keyword フラグを抽出
- `CardEditor.on_structure_update()` (line 217): structure_update_requested シグナルを処理
- `BaseEditForm.load_data()`: `_load_ui_from_data()` hook メソッドを呼び出し
- `BaseEditForm.save_data()`: `_save_ui_to_data()` hook メソッドを呼び出し

