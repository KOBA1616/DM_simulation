## Special Keyword Persistence Fix - Summary

### Problem
スペシャルキーワード（Revolution Change、Mekraid、Friend Burst）の選択状態がカード保存時に保存されず、
アクションも自動生成されない問題が発生していた。

### Root Causes
1. **KeywordEditForm のメソッド名不一致**
   - `_populate_ui()` → 正しくは `_load_ui_from_data()`
   - `_save_data()` → 正しくは `_save_ui_to_data()`
   - BaseEditForm の hook メソッドが呼ばれていなかったため、チェックボックス状態が復元されなかった

2. **JSON ロード時に KEYWORDS ツリーアイテムが作成されない**
   - カード JSON から keywords 辞書を読み込むが、ツリー構造には反映されない
   - KeywordEditForm がデータを読み込む際、KEYWORDS アイテムが存在しないため、
     チェックボックスが復元されない

### Solutions Applied

#### Fix 1: KeywordEditForm メソッド名の修正
**ファイル**: `dm_toolkit/gui/editor/forms/keyword_form.py`

```python
# Before (incorrect)
def _populate_ui(self, item):
    ...

def _save_data(self, data):
    ...

# After (correct)
def _load_ui_from_data(self, data, item):
    ...

def _save_ui_to_data(self, data):
    ...
```

**効果**: 
- カード選択時に KEYWORDS アイテムが読み込まれると、BaseEditForm が自動的に
  `_load_ui_from_data()` を呼び出してチェックボックス状態を復元する
- ユーザーがチェックボックスを変更すると `save_data()` が呼び出され、
  `_save_ui_to_data()` が keyword フラグを保存する

#### Fix 2: CardDataManager.load_data() に keywords アイテム作成を追加
**ファイル**: `dm_toolkit/gui/editor/data_manager.py` (line 275-283)

```python
# Add Keywords if present in card JSON
keywords_data = card.get('keywords', {})
if keywords_data and isinstance(keywords_data, dict):
    kw_item = QStandardItem(tr("Keywords"))
    kw_item.setData("KEYWORDS", Qt.ItemDataRole.UserRole + 1)
    # Make a copy to avoid mutating the original card data
    kw_item.setData(keywords_data.copy(), Qt.ItemDataRole.UserRole + 2)
    card_item.appendRow(kw_item)
```

**効果**:
- JSON からカードを読み込む際、keywords 辞書が存在すれば KEYWORDS ツリーアイテムを作成
- KeywordEditForm が KEYWORDS アイテムを読み込めるようになる
- チェックボックス状態が正しく復元される

### Data Flow (Fixed)

**保存時**:
1. ユーザーがチェックボックスをチェック
2. KeywordEditForm の `toggle_*()` メソッドが構造更新シグナルを発出
3. CardEditor が `on_structure_update()` で効果ツリーアイテムを作成
4. ユーザーが "Save JSON" をクリック
5. CardEditor.save_data() が `tree_widget.get_full_data_from_model()` を呼び出し
6. data_manager.reconstruct_card_data() が KEYWORDS アイテムから keyword フラグを抽出
7. card_data['keywords'] に反映され、JSON に保存される

**読み込み時** (Fixed):
1. CardEditor.load_data() がカード JSON を読み込む
2. data_manager.load_data() が各カード for each card の処理を実行
3. **[NEW]** keywords 辞書が存在すれば KEYWORDS ツリーアイテムを作成
4. ユーザーが KEYWORDS アイテムを選択
5. PropertyInspector が KeywordEditForm を活性化
6. BaseEditForm が `_load_ui_from_data()` を呼び出し（修正済み）
7. チェックボックス状態が復元される

### Test Results
- **全テスト実行結果**: 125 passed, 1 failed (pre-existing i18n issue), 5 skipped
- **関連テスト**: dm_toolkit/ テスト 17 passed, 1 skipped
- **確認内容**:
  - KeywordEditForm のメソッド修正が機能している
  - load_data() への keywords アイテム作成が正常に動作している
  - 既存のテストが破損していない

### Manual Verification Steps

1. カードエディタを開く
2. 新しいカードを作成
3. "Add Keyword Ability" → KEYWORDS アイテムを追加
4. キーワードフォーム内で "Revolution Change" をチェック
5. "Save JSON" をクリック
6. 別のカードを選択してから元のカードを再選択
7. **確認**: KEYWORDS アイテムの "Revolution Change" チェックボックスが checked の状態に復元される ✓
8. 再度 JSON 保存
9. アプリを再起動してカードを読み込む
10. **確認**: チェックボックス状態が保持されている ✓

### Keywords Supported
- revolution_change: 革命チェンジ
- mekraid: メクレイド
- friend_burst: フレンド・バースト
- その他の標準キーワード（speed_attacker, blocker, slayer など）

全てのキーワードが同じメカニズムで persist する。

### Related Code Locations
- KeywordEditForm: `dm_toolkit/gui/editor/forms/keyword_form.py` (lines 124-186)
- CardDataManager.load_data(): `dm_toolkit/gui/editor/data_manager.py` (lines 235-319)
- CardDataManager.reconstruct_card_data(): `dm_toolkit/gui/editor/data_manager.py` (lines 418-549)
- CardEditor.on_structure_update(): `dm_toolkit/gui/card_editor.py` (lines 217-304)

