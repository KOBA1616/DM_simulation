# メガラストバースト テキスト生成実装完了

## 📋 実装概要

ユーザー要求: メガラストバースト（Mega Last Burst）の CAST_SPELL テキスト生成時に、冒頭に「このクリーチャーがバトルゾーンから離れて」を追加する

**実装内容**: 
- `text_generator.py` の CAST_SPELL テキスト生成に mega_last_burst フラグ検出ロジックを追加
- カード全体の keywords から mega_last_burst フラグを検出し、エフェクト処理に伝播

## ✅ テスト結果

### テスト1: CAST_SPELL単独テキスト生成
```
条件: 
  - CAST_SPELL コマンド
  - card_mega_last_burst = False

結果:
  生成テキスト: 「呪文をコストを支払わずに唱える。」
  ✓ メガラストバースト接頭詞なし
```

### テスト2: メガラストバースト時のCASTSPELL
```
条件:
  - CAST_SPELL コマンド
  - card_mega_last_burst = True

結果:
  ✓ 生成テキスト: 「このクリーチャーがバトルゾーンから離れて、呪文をコストを支払わずに唱える。」
  ✓ 接頭詞が正しく追加
```

### テスト3: フルカード統合テスト
```
条件:
  - カードタイプ: CREATURE
  - キーワード: mega_last_burst = True
  - エフェクト: ON_DESTROY トリガー
  - コマンド: CAST_SPELL

結果:
  ✓ テキスト内に「このクリーチャーがバトルゾーンから離れて」が含まれる
  ✓ テキスト内に「唱える」が含まれる
  ✓ すべてのチェックが通過
```

## 🔧 実装変更内容

### 1. `_format_command()` メソッド修正

**ファイル**: `dm_toolkit/gui/editor/text_generator.py`  
**行範囲**: 778-817

**変更**: `card_mega_last_burst` パラメータを追加

```python
@classmethod
def _format_command(cls, command: Dict[str, Any], is_spell: bool = False, 
                   sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
```

**効果**: メガラストバーストフラグを action_proxy に設定

```python
action_proxy = {
    ...
    "is_mega_last_burst": card_mega_last_burst,  # ← 新規追加
    ...
}
```

### 2. `_format_effect()` メソッド修正

**ファイル**: `dm_toolkit/gui/editor/text_generator.py`  
**行番号**: 622行目

**変更**: `card_mega_last_burst` パラメータを追加、_format_command へ伝播

```python
@classmethod
def _format_effect(cls, effect: Dict[str, Any], is_spell: bool = False, 
                  sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
    ...
    # _format_command の呼び出しで mega_last_burst フラグを伝播
    action_texts.append(cls._format_command(command, is_spell, sample=sample, 
                                            card_mega_last_burst=card_mega_last_burst))
```

### 3. `generate_body_text()` メソッド修正

**ファイル**: `dm_toolkit/gui/editor/text_generator.py`  
**行番号**: 225-231行目

**変更**: エフェクト処理時にカードの mega_last_burst キーワードを検出して伝播

```python
for effect in effects:
    if _is_special_only_effect(effect):
        continue
    # Check if this card has mega_last_burst keyword and pass it to _format_effect
    has_mega_last_burst = data.get("keywords", {}).get("mega_last_burst", False)
    text = cls._format_effect(effect, is_spell, sample=sample, 
                             card_mega_last_burst=has_mega_last_burst)
    if text:
        lines.append(f"■ {text}")
```

### 4. CAST_SPELL テキスト生成ロジック修正

**ファイル**: `dm_toolkit/gui/editor/text_generator.py`  
**行番号**: 1734-1804

**変更**: mega_last_burst フラグを検出し、プレフィックスを生成

```python
elif atype == "CAST_SPELL":
    # ... (既存コード)
    
    # Mega Last Burst detection: check for mega_last_burst flag in context or action
    is_mega_last_burst = action.get("is_mega_last_burst", False) or action.get("mega_last_burst", False)
    mega_burst_prefix = ""
    if is_mega_last_burst:
        mega_burst_prefix = "このクリーチャーがバトルゾーンから離れて、"
    
    # ... (既存テンプレート生成)
    # 全テンプレートに mega_burst_prefix を追加
    template = f"{mega_burst_prefix}...テンプレート..."
```

## 📊 処理フロー

```
カード生成テキスト要求
  ↓
generate_body_text(card_data)
  ├─ card_data から mega_last_burst キーワード検出
  │  has_mega_last_burst = card_data.get("keywords", {}).get("mega_last_burst", False)
  │
  ├─ effect ループ
  │  └─ _format_effect() に has_mega_last_burst を渡す
  │
  ↓
_format_effect(effect, ..., card_mega_last_burst=True/False)
  │
  ├─ commands ループ
  │  └─ _format_command() に card_mega_last_burst を渡す
  │
  ↓
_format_command(command, ..., card_mega_last_burst=True/False)
  │
  ├─ command type が CAST_SPELL?
  │  └─ action_proxy に is_mega_last_burst を設定
  │
  ├─ _format_action(action_proxy) を呼び出し
  │
  ↓
_format_action(action_proxy)
  │
  ├─ atype == "CAST_SPELL" 判定
  │  └─ is_mega_last_burst = action.get("is_mega_last_burst", False)
  │
  ├─ mega_burst_prefix 生成
  │  └─ "このクリーチャーがバトルゾーンから離れて、" (if is_mega_last_burst)
  │
  ├─ テンプレート生成
  │  └─ template = f"{mega_burst_prefix}{テンプレート}"
  │
  ↓
生成テキスト: 「このクリーチャーがバトルゾーンから離れて、呪文をコストを支払わずに唱える。」
```

## ✨ 生成テキスト例

### パターン1: 基本的なメガラストバースト
```json
{
  "type": "CREATURE",
  "keywords": {"mega_last_burst": true},
  "effects": [
    {
      "trigger": "ON_DESTROY",
      "commands": [
        {
          "type": "CAST_SPELL",
          "target_filter": {"types": ["SPELL"]}
        }
      ]
    }
  ]
}
```
**生成テキスト**: 「このクリーチャーがバトルゾーンから離れて、呪文をコストを支払わずに唱える。」

### パターン2: 特定の呪文タイプ指定
```json
{
  "keywords": {"mega_last_burst": true},
  "effects": [
    {
      "trigger": "ON_DESTROY",
      "commands": [
        {
          "type": "CAST_SPELL",
          "target_filter": {
            "types": ["SPELL"],
            "civilizations": ["FIRE"]
          }
        }
      ]
    }
  ]
}
```
**生成テキスト**: 「このクリーチャーがバトルゾーンから離れて、火の呪文をコストを支払わずに唱える。」

### パターン3: ゾーン指定
```json
{
  "keywords": {"mega_last_burst": true},
  "effects": [
    {
      "trigger": "ON_DESTROY",
      "commands": [
        {
          "type": "CAST_SPELL",
          "target_filter": {
            "types": ["SPELL"],
            "zones": ["GRAVEYARD"]
          }
        }
      ]
    }
  ]
}
```
**生成テキスト**: 「このクリーチャーがバトルゾーンから離れて、墓地から呪文をコストを支払わずに唱える。」

## 🎯 整合性検証

| 項目 | 状態 | 詳細 |
|------|------|------|
| メガラストバースト検出 | ✅ | card keywords から正しく検出 |
| フラグ伝播 | ✅ | generate_body_text → _format_effect → _format_command → _format_action |
| プレフィックス生成 | ✅ | "このクリーチャーがバトルゾーンから離れて、" |
| テキスト統合 | ✅ | 複数エフェクト時も正しくマージ |
| 非メガラストバースト時 | ✅ | フラグなしで通常テキスト生成 |

## 🚀 デプロイ確認

- ✅ コード変更: `text_generator.py` を修正
- ✅ パラメータ追加:  
  - `_format_command()`: `card_mega_last_burst` パラメータ
  - `_format_effect()`: `card_mega_last_burst` パラメータ
- ✅ テキスト生成ロジック: メガラストバースト検出とプレフィックス追加
- ✅ テスト: 単独テスト・統合テストともにパス
- ✅ 後方互換性: パラメータがデフォルト False でメガラストバースト検出なし

---

**実装日時**: 2026年1月17日  
**ステータス**: ✅ 完了  
**レビュー**: 待機中
