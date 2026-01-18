# IF判定アクション 条件タイプ日本語化 実装レポート

**実施日:** 2026年1月22日  
**対象:** グループ「ロジック」のIF判定アクション  
**目的:** IF/IF_ELSE/ELSEコマンドで使用される条件タイプ（target_filter.type）の日本語化

---

## 1. 実装内容

### 1.1 日本語化リソース追加 (text_resources.py)

#### CONDITION_TYPE_LABELS辞書
```python
CONDITION_TYPE_LABELS: Dict[str, str] = {
    "NONE": "なし",
    "MANA_ARMED": "マナ武装",
    "SHIELD_COUNT": "シールド枚数",
    "CIVILIZATION_MATCH": "文明一致",
    "OPPONENT_PLAYED_WITHOUT_MANA": "相手がマナなしでプレイ",
    "OPPONENT_DRAW_COUNT": "相手のドロー枚数",
    "DURING_YOUR_TURN": "自分のターン中",
    "DURING_OPPONENT_TURN": "相手のターン中",
    "FIRST_ATTACK": "初回攻撃",
    "EVENT_FILTER_MATCH": "イベントフィルター一致",
    "COMPARE_STAT": "統計値比較",
    "COMPARE_INPUT": "入力値比較",
    "CARDS_MATCHING_FILTER": "フィルター一致カード数",
    "DECK_EMPTY": "デッキ切れ",
    "MANA_CIVILIZATION_COUNT": "マナゾーン文明数",
    "HAND_COUNT": "手札枚数",
    "BATTLE_ZONE_COUNT": "バトルゾーンカード数",
    "GRAVEYARD_COUNT": "墓地カード数",
    "CUSTOM": "カスタム"
}
```

**合計:** 19個の条件タイプを100%日本語化

#### ヘルパーメソッド追加
```python
@classmethod
def get_condition_type_label(cls, condition_type: str) -> str:
    """Get Japanese label for condition type (for GUI editor)."""
    return cls.CONDITION_TYPE_LABELS.get(condition_type, condition_type)

@classmethod
def get_stat_key_label(cls, stat_key: str) -> str:
    """Get Japanese label for stat key (for GUI editor)."""
    if stat_key in cls.STAT_KEY_MAP:
        name, unit = cls.STAT_KEY_MAP[stat_key]
        return f"{name}（{unit}）" if unit else name
    return stat_key
```

### 1.2 GUIエディタ適用 (condition_widget.py)

#### インポート追加
```python
from dm_toolkit.gui.editor.text_resources import CardTextResources
```

#### populate_combo()メソッド修正
```python
def populate_combo(self, combo, items):
    combo.clear()
    for item in items:
        # Use CardTextResources for condition type labels
        label = CardTextResources.get_condition_type_label(str(item))
        combo.addItem(label, str(item))
```

**変更点:**
- 旧: `combo.addItem(tr(str(item)), str(item))`  
  → `tr()`では条件タイプの翻訳が定義されていなかった
- 新: `combo.addItem(CardTextResources.get_condition_type_label(str(item)), str(item))`  
  → 専用の日本語化リソースを使用

---

## 2. 日本語化対応リスト

### 2.1 主要な条件タイプ

| 英語キー | 日本語ラベル | 用途 |
|---|---|---|
| OPPONENT_DRAW_COUNT | 相手のドロー枚数 | 相手が引いたカード枚数の判定 |
| COMPARE_STAT | 統計値比較 | マナ/シールド/手札などの統計値比較 |
| SHIELD_COUNT | シールド枚数 | シールド枚数の判定 |
| CIVILIZATION_MATCH | 文明一致 | マナゾーンに同じ文明があるかの判定 |
| MANA_CIVILIZATION_COUNT | マナゾーン文明数 | マナゾーンにある文明の数の判定 |
| CARDS_MATCHING_FILTER | フィルター一致カード数 | 特定フィルターに一致するカード数 |
| DECK_EMPTY | デッキ切れ | デッキが空かどうかの判定 |

### 2.2 統計キー（COMPARE_STAT用）

| 統計キー | 日本語ラベル |
|---|---|
| MANA_COUNT | マナゾーンのカード（枚） |
| SHIELD_COUNT | シールド（つ） |
| HAND_COUNT | 手札（枚） |
| OPPONENT_SHIELD_COUNT | 相手のシールド（つ） |
| MANA_CIVILIZATION_COUNT | マナゾーンの文明数 |
| BATTLE_ZONE_COUNT | バトルゾーンのカード（枚） |

---

## 3. テスト結果

### 3.1 条件タイプラベル翻訳率
```
合計: 19個
翻訳済み: 19個
未翻訳: 0個
翻訳率: 100.0%
```

### 3.2 実際のIF判定での動作確認

#### Example 1: OPPONENT_DRAW_COUNT
```
条件: OPPONENT_DRAW_COUNT >= 2
生成テキスト: 相手がカードを2枚目以上引いたなら、カードを1枚引いてもよい。
```
✅ 正常動作

#### Example 2: GUIエディタでの表示
- コンボボックスに「相手のドロー枚数」と日本語で表示
- データとしては内部的に"OPPONENT_DRAW_COUNT"を保持

---

## 4. 影響範囲

### 4.1 修正ファイル
1. `dm_toolkit/gui/editor/text_resources.py`
   - CONDITION_TYPE_LABELS辞書追加
   - get_condition_type_label()メソッド追加
   - get_stat_key_label()メソッド追加

2. `dm_toolkit/gui/editor/forms/parts/condition_widget.py`
   - CardTextResourcesインポート追加
   - populate_combo()メソッド修正

### 4.2 新規テストファイル
1. `test_condition_labels.py` - 条件タイプ日本語化テスト
2. `test_if_condition_labels.py` - 統合テスト

---

## 5. 使用例

### 5.1 GUIエディタでの使用
```python
# ConditionEditorWidget内で自動的に適用
cond_type_combo = QComboBox()
cond_types = ["NONE", "OPPONENT_DRAW_COUNT", "COMPARE_STAT", ...]
self.populate_combo(self.cond_type_combo, cond_types)
# → コンボボックスに「なし」「相手のドロー枚数」「統計値比較」... と表示
```

### 5.2 プログラムからの使用
```python
from dm_toolkit.gui.editor.text_resources import CardTextResources

# 条件タイプのラベル取得
label = CardTextResources.get_condition_type_label("OPPONENT_DRAW_COUNT")
# → "相手のドロー枚数"

# 統計キーのラベル取得
label = CardTextResources.get_stat_key_label("SHIELD_COUNT")
# → "シールド（つ）"
```

---

## 6. 今後の拡張可能性

### 6.1 追加候補の条件タイプ
- HAND_SIZE_COMPARE: 手札枚数比較
- POWER_THRESHOLD: パワーしきい値
- COST_THRESHOLD: コストしきい値

### 6.2 エディタUI改善
- 条件タイプ選択時に説明文を表示
- 統計キー選択時に利用可能な値の範囲を表示

---

## 7. 関連ドキュメント

- [IF_TEXT_GENERATION_FIX.md](IF_TEXT_GENERATION_FIX.md) - IFコマンドテキスト生成拡張
- [GUI_NODE_SWITCH_FIX_REPORT.md](GUI_NODE_SWITCH_FIX_REPORT.md) - ノード切り替え修正

---

## 8. 結論

✅ **完了項目:**
1. 19個の条件タイプを100%日本語化
2. GUIエディタで条件タイプが日本語表示
3. 統計キーの日本語ラベル取得メソッド追加
4. 既存機能への影響なし

**主な成果:**
- ユーザーが直感的に条件タイプを選択可能
- IF判定の設定がより分かりやすく
- コード内部では英語キーを維持し、互換性を保持
