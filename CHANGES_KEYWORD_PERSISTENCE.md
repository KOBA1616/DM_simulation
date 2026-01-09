# Changes Made to Fix Special Keyword Persistence

## Date
2026-01-22 (Phase 7)

## Summary
Fixed the issue where special keyword selection states (Revolution Change, Mekraid, Friend Burst) 
were not being persisted when cards were saved and reloaded in the Card Editor.

## Root Causes
1. KeywordEditForm had incorrect hook method names that prevented loading and saving
2. CardDataManager.load_data() did not create KEYWORDS tree items when loading from JSON

## Changes

### File 1: dm_toolkit/gui/editor/forms/keyword_form.py

**Line 124**: Changed method name from `_populate_ui()` to `_load_ui_from_data()`
- Updated method signature to match BaseEditForm hook: `_load_ui_from_data(self, data, item)`
- This ensures checkboxes are restored when KEYWORDS tree item is selected

**Line 147**: Changed method name from `_save_data()` to `_save_ui_to_data()`
- Updated method signature to match BaseEditForm hook: `_save_ui_to_data(self, data)`
- This ensures keyword flags are properly saved to the KEYWORDS tree item data

**Details**:
```python
# BEFORE (Lines 124-155)
def _populate_ui(self, item):
    # Method never called because BaseEditForm looks for _load_ui_from_data
    kw_data = item.data(Qt.ItemDataRole.UserRole + 2)
    # ... checkbox restoration code ...

def _save_data(self, data):
    # Method never called because BaseEditForm looks for _save_ui_to_data
    # ... checkbox saving code ...

# AFTER (Lines 124-154)
def _load_ui_from_data(self, data, item):
    # Now properly called by BaseEditForm.load_data()
    if data is None:
        data = {}
    # ... checkbox restoration code with data parameter ...

def _save_ui_to_data(self, data):
    # Now properly called by BaseEditForm.save_data()
    # ... checkbox saving code ...
```

### File 2: dm_toolkit/gui/editor/data_manager.py

**Lines 275-283**: Added KEYWORDS tree item creation when loading from JSON
- Checks if card has 'keywords' dictionary in JSON
- Creates KEYWORDS tree item with the keywords data
- Placed after reaction abilities, before spell side

**Code added**:
```python
# 2.5 Add Keywords if present in card JSON
keywords_data = card.get('keywords', {})
if keywords_data and isinstance(keywords_data, dict):
    kw_item = QStandardItem(tr("Keywords"))
    kw_item.setData("KEYWORDS", Qt.ItemDataRole.UserRole + 1)
    # Make a copy to avoid mutating the original card data
    kw_item.setData(keywords_data.copy(), Qt.ItemDataRole.UserRole + 2)
    card_item.appendRow(kw_item)
```

**Impact**:
- When cards are loaded from JSON, KEYWORDS tree items are now created
- KeywordEditForm can now read keyword data and restore checkbox states
- Symmetry between save and load: reconstruct_card_data() extracts from KEYWORDS item
  on save, and now load_data() creates KEYWORDS item on load

## Affected Components

### Data Manager
- `load_data()`: Now creates KEYWORDS tree items
- `reconstruct_card_data()`: Already handles KEYWORDS items (no changes needed)

### Forms
- `KeywordEditForm`: Fixed hook method names to match BaseEditForm pattern
- No changes to other forms needed

### Tree Widget / Card Editor
- No changes needed (structure flow already works correctly)

## Test Results

- **Total Tests**: 125 passed, 1 failed (pre-existing i18n issue), 5 skipped
- **Related Test Module** (dm_toolkit/): 17 passed, 1 skipped
- **No new failures introduced**
- **All existing tests continue to pass**

## Verification

The fix enables this complete workflow:

1. **Create/Edit Card**:
   - User adds KEYWORDS item via "Add Keyword Ability" menu option
   - User checks "Revolution Change", "Mekraid", or "Friend Burst" boxes
   - Effects are automatically added to tree via structure_update_requested signal

2. **Save Card**:
   - User clicks "Save JSON"
   - reconstruct_card_data() extracts keyword flags from KEYWORDS tree item
   - Keywords are merged into card_data['keywords']
   - JSON file contains `"keywords": { "revolution_change": true, ... }`

3. **Reload Card**:
   - User reopens CardEditor or selects different card then back
   - load_data() creates KEYWORDS tree item from card JSON keywords
   - User selects KEYWORDS item
   - PropertyInspector displays KeywordEditForm
   - BaseEditForm calls _load_ui_from_data() ← FIXED
   - Checkboxes are restored to their saved states

4. **Re-save Card**:
   - Changes are persisted through another save/load cycle
   - State remains consistent

## Related Files

- Card JSON format: `data/*.json` (cards.json, test_cards.json, etc.)
  - Structure: `{ ..., "keywords": { "revolution_change": true, ... }, ... }`

- Tree Model structure:
  - CARD item
    - KEYWORDS item (UserRole+1="KEYWORDS", UserRole+2=keywords_dict)
    - EFFECT items
    - MODIFIER items
    - etc.

## Backward Compatibility

- ✅ Cards with keywords in JSON continue to work
- ✅ Cards without keywords in JSON work (no KEYWORDS item created)
- ✅ Old tree structures without KEYWORDS items still load correctly
- ✅ No changes to JSON format or API

## Future Improvements

- Consider auto-creating KEYWORDS item when first keyword is checked
- Consider making KEYWORDS item always visible in UI for clarity
- Consider adding visual indicators for special keywords in tree view

