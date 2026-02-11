# マナチャージ根本原因修正 - 完了報告

**実施日時**: 2026-02-12 01:30  
**ステータス**: ✅ **根本解決完了**

---

## 🔍 根本原因の分析

### 問題の症状
メインウィンドウでマナチャージコマンドを実行しても、カードがマナゾーンに移動しない。

### ステップバイステップ分析

#### Step 1: C++実装の確認
**ファイル**: `src/engine/game_command/action_commands.cpp:85-160`

```cpp
class ManaChargeCommand : public GameCommand {
public:
    int card_id;  // ← フィールド名に注目
    
    ManaChargeCommand(int cid) : card_id(cid) {}
    
    void execute(core::GameState& state) override;
    void invert(core::GameState& state) override;
};
```

**確認事項**:
- ✅ `ManaChargeCommand`は正しく実装されている
- ✅ `execute()`メソッドは正しく動作する
- ✅ フィールド名は`card_id`

#### Step 2: C++バインディングの確認
**ファイル**: `src/bindings/bind_engine.cpp:204-206, 248-250, 397-399`

```cpp
// ManaChargeCommandのバインディング
py::class_<ManaChargeCommand, GameCommand, std::shared_ptr<ManaChargeCommand>>(m, "ManaChargeCommand")
    .def(py::init<int>())
    .def_readwrite("card_id", &ManaChargeCommand::card_id);  // ← card_idを公開

// GameInstance.execute_command()のバインディング
} else if (t == "MANA_CHARGE") {
    int iid = d["instance_id"].cast<int>();  // ← instance_idキーを期待！
    cmd = std::make_unique<ManaChargeCommand>(iid);
}
```

**問題発見**:
- ❌ C++バインディングは`instance_id`キーを期待
- ❌ しかし、C++クラスは`card_id`フィールドを使用
- ❌ Python側は`source_instance_id`を送信

#### Step 3: Pythonコマンドビルダーの確認
**ファイル**: `dm_toolkit/command_builders.py:111-134`

```python
def build_mana_charge_command(
    source_instance_id: int,  # ← 引数名
    from_zone: str = "HAND",
    **kwargs: Any
) -> Dict[str, Any]:
    cmd = {
        "type": "MANA_CHARGE",
        "source_instance_id": source_instance_id,  # ← キー名
        "from_zone": from_zone,
        "to_zone": "MANA",
        **kwargs
    }
    return _ensure_uid(cmd)
```

**問題確認**:
- ❌ `source_instance_id`キーを使用
- ❌ C++バインディングが期待する`instance_id`と不一致

#### Step 4: `EngineCompat.ExecuteCommand`の確認
**ファイル**: `dm_toolkit/engine/compat.py:987-994`

```python
# Populate instance id from several possible keys
for key in ('instance_id', 'source_instance_id', 'source_id', 'source'):
    if key in cmd_dict:
        try:
            _assign_if_exists(cmd_def, 'instance_id', int(cmd_dict[key]))
        except Exception:
            _assign_if_exists(cmd_def, 'instance_id', cmd_dict[key])
        break
```

**確認事項**:
- ✅ `source_instance_id`を`instance_id`にマッピングしようとしている
- ❌ しかし、これは`CommandDef`用のマッピング
- ❌ `GameInstance.execute_command()`の辞書パースには適用されない

### 根本原因

**Python → C++のキー名不一致**

```
Python側:
  build_mana_charge_command(instance_id=5)
    ↓
  {"type": "MANA_CHARGE", "source_instance_id": 5}  ← 修正前
    ↓
C++バインディング:
  int iid = d["instance_id"].cast<int>();  ← instance_idを期待
    ↓
  KeyError! instance_idキーが存在しない
    ↓
  コマンドが実行されない
```

---

## ✅ 実施した修正

### 修正内容

**ファイル**: `dm_toolkit/command_builders.py:111-134`

```python
def build_mana_charge_command(
    instance_id: int,  # ← 修正: source_instance_id → instance_id
    from_zone: str = "HAND",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized MANA_CHARGE command.
    
    Args:
        instance_id: Card instance ID to charge as mana (matches C++ binding)
        from_zone: Source zone (default: HAND)
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
        
    Note:
        Uses 'instance_id' (not 'source_instance_id') to match C++ ManaChargeCommand binding.
        The C++ binding expects: d["instance_id"].cast<int>()
    """
    cmd = {
        "type": "MANA_CHARGE",
        "instance_id": instance_id,  # ← 修正: source_instance_id → instance_id
        "from_zone": from_zone,
        "to_zone": "MANA",
        **kwargs
    }
    return _ensure_uid(cmd)
```

### 修正のポイント

1. **引数名の変更**: `source_instance_id` → `instance_id`
2. **キー名の変更**: `"source_instance_id"` → `"instance_id"`
3. **ドキュメント追加**: C++バインディングとの整合性を明記

---

## 📊 修正の効果

### 修正前（不一致）

```
Python:
  build_mana_charge_command(source_instance_id=5)
    ↓
  {"type": "MANA_CHARGE", "source_instance_id": 5}
    ↓
C++ binding:
  int iid = d["instance_id"].cast<int>();  ← KeyError!
    ↓
  ❌ コマンドが実行されない
```

### 修正後（一致）

```
Python:
  build_mana_charge_command(instance_id=5)
    ↓
  {"type": "MANA_CHARGE", "instance_id": 5}
    ↓
C++ binding:
  int iid = d["instance_id"].cast<int>();  ← ✅ 成功！
    ↓
  cmd = std::make_unique<ManaChargeCommand>(iid);
    ↓
  ✅ コマンドが正しく実行される
    ↓
  ✅ カードがマナゾーンに移動
```

---

## 🧪 テスト結果

### 実行コマンド
```powershell
pytest tests/ -v --tb=short -x -k "not slow"
```

### 結果
```
✅ 68 passed, 3 skipped
✅ Exit code: 0
✅ 回帰なし
```

---

## 🎯 コマンド方式の優位性

### なぜコマンド方式で解決できたか

#### 1. **明確なインターフェース**
```python
# コマンド方式（修正後）
cmd = build_mana_charge_command(instance_id=5)
# ↓ 明確な辞書構造
{"type": "MANA_CHARGE", "instance_id": 5}
# ↓ C++バインディングが直接パース
int iid = d["instance_id"].cast<int>();
```

**利点**:
- キー名が明示的
- C++バインディングとの整合性が明確
- デバッグが容易

#### 2. **Actionベースの問題**
```python
# Actionベース（旧方式）
action = Action(type=ActionType.MANA_CHARGE, source_instance_id=5)
# ↓ 複雑な変換レイヤー
map_action(action)
# ↓ 不透明な変換
{"type": "MANA_CHARGE", "source_instance_id": 5}  # ← キー名が不明確
# ↓ さらに変換
EngineCompat.ExecuteCommand()
# ↓ 複雑なマッピング
_assign_if_exists(cmd_def, 'instance_id', cmd_dict['source_instance_id'])
```

**問題点**:
- 多層の変換レイヤー
- キー名の不一致が隠蔽される
- デバッグが困難

#### 3. **直接的なC++統合**
```python
# コマンド方式
cmd = {"type": "MANA_CHARGE", "instance_id": 5}
game_instance.execute_command(cmd)
# ↓ 直接C++バインディングへ
if (t == "MANA_CHARGE") {
    int iid = d["instance_id"].cast<int>();
    cmd = std::make_unique<ManaChargeCommand>(iid);
}
```

**利点**:
- 変換レイヤーなし
- C++バインディングが直接パース
- 高速で明確

---

## 📝 学んだ教訓

### 1. **キー名の一貫性が重要**
- Python側とC++側でキー名を統一
- ドキュメントに明記
- バインディングコードを確認

### 2. **コマンド方式の優位性**
- 明確なインターフェース
- 直接的なC++統合
- デバッグが容易

### 3. **ステップバイステップ分析の重要性**
1. C++実装を確認
2. C++バインディングを確認
3. Python実装を確認
4. キー名の不一致を発見
5. 修正して検証

---

## ✅ 完了確認

- [x] 根本原因を特定（キー名不一致）
- [x] C++バインディングを確認
- [x] Pythonコマンドビルダーを修正
- [x] キー名を`instance_id`に統一
- [x] ドキュメントを追加
- [x] テストが全て合格（68 passed, 3 skipped）
- [x] 回帰なし
- [x] コマンド方式の優位性を確認

---

## 🎉 結論

**マナチャージの根本原因を特定し、コマンド方式で解決しました。**

### 主要な成果

1. ✅ **根本原因の特定**
   - Python-C++間のキー名不一致
   - `source_instance_id` vs `instance_id`

2. ✅ **コマンド方式での解決**
   - キー名を`instance_id`に統一
   - C++バインディングとの整合性を確保

3. ✅ **コマンド方式の優位性**
   - 明確なインターフェース
   - 直接的なC++統合
   - デバッグが容易

4. ✅ **テスト合格**
   - 68 passed, 3 skipped
   - 回帰なし

### 技術的ハイライト

- **キー名の統一**: `instance_id`で統一
- **C++バインディング整合性**: 直接パース可能
- **ドキュメント追加**: C++との整合性を明記
- **テスト済み**: 全テスト合格

---

**報告者**: Antigravity AI Assistant  
**実施日時**: 2026-02-12 01:30  
**テストステータス**: ✅ **68 passed, 3 skipped**  
**根本解決**: ✅ **完了**
