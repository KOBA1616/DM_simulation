# Scope and Keyword Ability Unification

## 概要

Static Ability（スタティック効果）とTrigger Effect（トリガー効果）における以下の2つの設計上の問題を解決しました：

1. **スコープ定数の統一**: `PLAYER_SELF`/`PLAYER_OPPONENT`と`SELF`/`OPPONENT`の表記揺れ
2. **キーワード能力の型安全性**: `str_val`フィールドの汎用性による曖昧さ

## 実装内容

### 1. TargetScope Enum（統一スコープ定数）

#### ファイル: `dm_toolkit/consts.py`

```python
class TargetScope:
    """Unified target scope constants."""
    SELF = "SELF"
    OPPONENT = "OPPONENT"
    ALL = "ALL"
    
    # Legacy aliases
    PLAYER_SELF = "SELF"
    PLAYER_OPPONENT = "OPPONENT"
    
    @classmethod
    def normalize(cls, value: str) -> str:
        """Normalize legacy PLAYER_* values to unified format."""
```

**用途**:
- Static Abilities: `scope`フィールド
- Trigger Effects: `target_group`フィールド
- Filter definitions: `owner`フィールド

**利点**:
- ✅ 一貫性のある命名規則
- ✅ 後方互換性の維持（`PLAYER_SELF` → `SELF`への自動変換）
- ✅ 型安全性の向上（`all_values()`で有効値を取得可能）

### 2. mutation_kind フィールド（キーワード専用フィールド）

#### Static Ability構造の変更

**従来**:
```json
{
  "type": "GRANT_KEYWORD",
  "str_val": "speed_attacker",  // 汎用フィールド（曖昧）
  "scope": "PLAYER_SELF"
}
```

**新設計**:
```json
{
  "type": "GRANT_KEYWORD",
  "mutation_kind": "speed_attacker",  // キーワード専用フィールド
  "str_val": "speed_attacker",  // 後方互換性のため保持
  "scope": "SELF"
}
```

**対象修飾子タイプ**:
- `GRANT_KEYWORD`: キーワード能力付与
- `SET_KEYWORD`: キーワード能力設定

**利点**:
- ✅ 型の明確化（キーワード ≠ メッセージ ≠ ファイルパス）
- ✅ バリデーション強化（`mutation_kind`の存在チェック）
- ✅ コード可読性の向上

### 3. データマイグレーション機能

#### ファイル: `dm_toolkit/gui/editor/data_migration.py`

**主要機能**:

1. **`migrate_modifier_keyword_field()`**
   - `str_val` → `mutation_kind`への自動移行
   - キーワードタイプ（GRANT_KEYWORD/SET_KEYWORD）のみ対象

2. **`normalize_modifier_scope()`**
   - `PLAYER_SELF` → `SELF`への正規化
   - `PLAYER_OPPONENT` → `OPPONENT`への正規化

3. **`migrate_card_data()`**
   - カード全体のstatic_abilitiesを一括移行
   - 移行済みアイテムはスキップ

4. **`verify_migration()`**
   - 移行漏れの検出
   - 警告メッセージの生成

**使用例**:
```python
from dm_toolkit.gui.editor.data_migration import migrate_card_data

# Load card JSON
card_data = load_card_json("some_card.json")

# Auto-migrate
migrated_count = migrate_card_data(card_data)
print(f"Migrated {migrated_count} fields")

# Save updated JSON
save_card_json("some_card.json", card_data)
```

### 4. 自動マイグレーション統合

#### ファイル: `dm_toolkit/gui/editor/data_manager.py`

`_normalize_card_for_engine()`メソッドに自動マイグレーションを統合：

```python
def _normalize_card_for_engine(self, card: dict) -> list:
    """Normalize card JSON to be engine-friendly..."""
    from dm_toolkit.gui.editor.data_migration import migrate_card_data
    
    # Auto-migrate card data
    migrated_count = migrate_card_data(card)
    if migrated_count > 0:
        warnings.append(f"自動マイグレーション: {migrated_count}個のフィールドを更新しました")
    # ... rest of normalization
```

**動作**:
- カード保存時に自動的に実行
- 警告表示により移行を通知
- ユーザー介入不要

### 5. UI更新

#### ファイル: `dm_toolkit/gui/editor/forms/modifier_form.py`

**読み込み時**（`_populate_ui_from_data()`）:
```python
# Prefer mutation_kind, fallback to str_val for legacy data
keyword = data.get('mutation_kind', '') or data.get('str_val', '')
if keyword:
    self.keyword_combo.set_keyword(keyword)
```

**保存時**（`_save_ui_to_data()`）:
```python
# Save keyword to both fields
if mtype in ('GRANT_KEYWORD', 'SET_KEYWORD'):
    keyword = self.keyword_combo.get_keyword()
    data['mutation_kind'] = keyword  # Primary field
    data['str_val'] = keyword  # Legacy support
```

**スコープ処理**:
```python
from dm_toolkit.consts import TargetScope

if self_checked and opp_checked:
    scope_value = TargetScope.ALL
elif self_checked:
    scope_value = TargetScope.SELF
# ...
data['scope'] = scope_value
```

#### ファイル: `dm_toolkit/gui/editor/forms/unified_widgets.py`

```python
from dm_toolkit.consts import TargetScope

def make_scope_combo(parent=None, include_zones=False):
    """Uses TargetScope.SELF/OPPONENT for player scopes."""
    combo = QComboBox(parent)
    scopes = [
        TargetScope.SELF,  # "SELF" instead of "PLAYER_SELF"
        TargetScope.OPPONENT,  # "OPPONENT"
        "TARGET_SELECT",
        # ...
    ]
```

### 6. バリデータ更新

#### ファイル: `dm_toolkit/gui/editor/validators_shared.py`

```python
class ModifierValidator:
    @staticmethod
    def validate(modifier: Dict[str, Any]) -> List[str]:
        # ...
        elif mtype in ["GRANT_KEYWORD", "SET_KEYWORD"]:
            # Check mutation_kind first (preferred), fallback to str_val (legacy)
            has_mutation_kind = 'mutation_kind' in modifier and modifier.get('mutation_kind')
            has_str_val = 'str_val' in modifier and modifier.get('str_val')
            
            if not has_mutation_kind and not has_str_val:
                errors.append(f"{mtype} requires 'mutation_kind' or 'str_val'")
        
        # Scope validation (using TargetScope)
        from dm_toolkit.consts import TargetScope
        scope = modifier.get('scope', TargetScope.ALL)
        scope = TargetScope.normalize(scope)
        if scope not in TargetScope.all_values():
            errors.append(f"Invalid scope: '{scope}'")
```

### 7. テキスト生成更新

#### ファイル: `dm_toolkit/gui/editor/text_generator.py`

```python
def _format_modifier(cls, modifier: Dict[str, Any], sample: List[Any] = None) -> str:
    from dm_toolkit.consts import TargetScope
    
    # Prefer mutation_kind, fallback to str_val for keywords
    keyword = modifier.get("mutation_kind", "") or modifier.get("str_val", "")
    
    # Normalize scope using TargetScope
    scope = modifier.get("scope", TargetScope.ALL)
    scope = TargetScope.normalize(scope)
    # ...
```

#### ファイル: `dm_toolkit/gui/editor/text_resources.py`

```python
from dm_toolkit.consts import TargetScope

class CardTextResources:
    SCOPE_JAPANESE: Dict[str, str] = {
        TargetScope.SELF: "自分の",
        TargetScope.OPPONENT: "相手の",
        TargetScope.ALL: "",
        # Legacy support
        "PLAYER_SELF": "自分の",
        "PLAYER_OPPONENT": "相手の",
    }
```

## テスト

### テストファイル: `python/tests/dm_toolkit/test_scope_mutation_kind.py`

**テストカバレッジ**:
- ✅ TargetScope定数（4テスト）
- ✅ mutation_kind移行（4テスト）
- ✅ スコープ正規化（3テスト）
- ✅ カード一括移行（1テスト）
- ✅ バリデータ統合（5テスト）
- ✅ 移行検証（3テスト）

**実行結果**:
```
20 passed in 0.16s
```

**既存テストとの互換性**:
```
python/tests/gui/test_validators_shared.py::TestModifierValidator
9 passed in 0.08s
```

## 後方互換性

### レガシーデータのサポート

1. **スコープフィールド**:
   - `PLAYER_SELF` → 自動的に`SELF`に正規化
   - `PLAYER_OPPONENT` → 自動的に`OPPONENT`に正規化
   - 既存データは破損しない

2. **str_valフィールド**:
   - `mutation_kind`不在時は`str_val`を使用
   - 両方存在する場合は`mutation_kind`を優先
   - 既存の`str_val`ベースのデータも正常動作

3. **自動マイグレーション**:
   - 保存時に自動実行
   - 手動介入不要
   - 警告メッセージで移行を通知

### 移行パス

**推奨手順**:
1. エディタでカードを開く
2. 修正不要（自動マイグレーション実行）
3. 保存ボタンをクリック
4. 警告メッセージ確認（例: "自動マイグレーション: 2個のフィールドを更新しました"）

## ベストプラクティス

### 新規コード

**スコープ指定**:
```python
from dm_toolkit.consts import TargetScope

modifier = {
    "type": "POWER_MODIFIER",
    "value": 2000,
    "scope": TargetScope.SELF,  # 推奨
    "condition": {"type": "NONE"},
    "filter": {}
}
```

**キーワード能力**:
```python
modifier = {
    "type": "GRANT_KEYWORD",
    "mutation_kind": "speed_attacker",  # 推奨
    "scope": TargetScope.SELF,
    "condition": {"type": "NONE"},
    "filter": {}
}
```

### レガシーコードの更新

**移行ツール使用**:
```python
from dm_toolkit.gui.editor.data_migration import verify_migration

# Check for migration issues
warnings = verify_migration(modifier)
if warnings:
    for w in warnings:
        print(f"Warning: {w}")
```

## 影響範囲

### 変更されたファイル

1. **Core Constants**:
   - `dm_toolkit/consts.py` - TargetScope定義

2. **Migration**:
   - `dm_toolkit/gui/editor/data_migration.py` - 新規作成

3. **Forms**:
   - `dm_toolkit/gui/editor/forms/modifier_form.py` - mutation_kind対応
   - `dm_toolkit/gui/editor/forms/unified_widgets.py` - TargetScope使用

4. **Validation**:
   - `dm_toolkit/gui/editor/validators_shared.py` - mutation_kind検証

5. **Text Generation**:
   - `dm_toolkit/gui/editor/text_generator.py` - mutation_kind優先読み取り
   - `dm_toolkit/gui/editor/text_resources.py` - TargetScope対応

6. **Data Management**:
   - `dm_toolkit/gui/editor/data_manager.py` - 自動マイグレーション統合

7. **Tests**:
   - `python/tests/dm_toolkit/test_scope_mutation_kind.py` - 新規作成

### 影響を受けないファイル

- Command実行ロジック（`action_to_command.py`等） - mutation_kindは既存サポート
- Engine側のC++コード - JSONフォーマットは互換性維持
- 既存カードデータファイル - 自動マイグレーション対応

## まとめ

この変更により以下が達成されました：

1. **設計の一貫性**: スコープ定数が統一され、コードベース全体で一貫した命名規則を使用
2. **型安全性の向上**: キーワード能力が専用フィールド（`mutation_kind`）で管理され、型の曖昧さが解消
3. **保守性の改善**: レガシーコードとの互換性を維持しつつ、段階的な移行が可能
4. **品質保証**: 20個の新規テストケースで動作を検証、既存テストも全てパス

**次のステップ**:
- 既存カードデータの一括移行スクリプト作成（オプション）
- エディタUIでの移行状況表示機能（オプション）
- レガシーサポートの段階的廃止計画（将来）
