# CAST_SPELL + REPLACE_CARD_MOVE テキスト生成統合完了

## 📋 実装概要

ユーザー要求: 呪文詠唱後に、墓地に置くかわりに山札の下に置く置換効果を表現するテキスト生成

**実装内容**: `text_generator.py` の `_merge_action_texts()` メソッドに CAST_SPELL + REPLACE_CARD_MOVE パターン対応を追加

## ✅ テスト結果

### テスト1: テキスト生成マージ
```
条件: 
  - コマンド1: CAST_SPELL
  - コマンド2: REPLACE_CARD_MOVE
    - from_zone: GRAVEYARD (墓地)
    - to_zone: DECK_BOTTOM (山札の下)
    - input_value_key: card_ref

結果:
  ✓ 生成テキスト: "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。"
  ✓ テキスト長: 27 文字
  ✓ 含まれる要素:
    - "呪文": ✓
    - "唱えた": ✓ 
    - "置くかわりに": ✓
    - "墓地": ✓
    - "山札の下": ✓
```

### テスト2: コマンド変換（エンジン互換性）
```
実行パス: ensure_executable_command() → map_action() → EngineCompat
状態: ✅ REPLACE_CARD_MOVE コマンドが正常に変換される
```

### テスト3: エンジン実行処理
```
場所: dm_toolkit/engine/compat.py:416
処理内容:
  1. instance_id からカードを取得
  2. from_zone (GRAVEYARD) から削除
  3. to_zone (DECK_BOTTOM) に配置
結果: ✅ Python フォールバック実装により正常に処理可能
```

## 🔧 実装変更内容

### ファイル: `dm_toolkit/gui/editor/text_generator.py`

#### 変更内容: `_merge_action_texts()` メソッドの拡張

**追加パターン検出関数**:
```python
def is_cast_spell_item(it):
    """CAST_SPELL コマンド判定"""
    if not isinstance(it, dict):
        return False
    t = it.get('type', '').upper()
    return t == 'CAST_SPELL'

def is_replace_card_move(it):
    """REPLACE_CARD_MOVE コマンド判定"""
    if not isinstance(it, dict):
        return False
    t = it.get('type', '').upper()
    return t == 'REPLACE_CARD_MOVE'
```

**マージロジック**:
```python
# Pattern: spell cast followed by replacement move
if len(raw_items) >= 2 and is_cast_spell_item(raw_items[0]) and is_replace_card_move(raw_items[1]):
    from_zone_text = tr(raw_items[1].get('from_zone', 'GRAVEYARD'))
    to_zone_text = tr(raw_items[1].get('to_zone', 'DECK_BOTTOM'))
    merged = f"その呪文を唱えた後、{from_zone_text}に置くかわりに{to_zone_text}に置く。"
```

## 📊 整合性検証

| 項目 | 状態 | 詳細 |
|------|------|------|
| テキスト生成 | ✅ | "その呪文を唱えた後、墓地に置くかわりに山札の下に置く。" |
| コマンド変換 | ✅ | map_action() で正常に REPLACE_CARD_MOVE に変換 |
| エンジン実行 | ✅ | compat.py で Python フォールバック実装あり |
| ゾーン翻訳 | ✅ | tr() で "墓地" "山札の下" に正しく翻訳 |
| マージタイミング | ✅ | コマンドシーケンス時に自動的にマージ |

## 🎯 生成テキスト例

### パターン1: 基本
```json
{
  "type": "CAST_SPELL",
  "target_group": "SELF"
}
{
  "type": "REPLACE_CARD_MOVE",
  "from_zone": "GRAVEYARD",
  "to_zone": "DECK_BOTTOM",
  "input_value_key": "card_ref"
}
```
**生成テキスト**: 「その呪文を唱えた後、墓地に置くかわりに山札の下に置く。」

### パターン2: 異なるゾーン指定
```json
{
  "type": "CAST_SPELL",
  "target_group": "SELF"
}
{
  "type": "REPLACE_CARD_MOVE",
  "from_zone": "HAND",
  "to_zone": "MANA_ZONE",
  "input_value_key": "card_ref"
}
```
**生成テキスト**: 「その呪文を唱えた後、手札に置くかわりにマナゾーンに置く。」

## 📝 処理フロー

```
UIエディタで設定
  ↓
CardTextGenerator._format_command()
  ├─ CAST_SPELL テキスト生成
  └─ REPLACE_CARD_MOVE テキスト生成
  ↓
CardTextGenerator._merge_action_texts()
  ├─ パターン検出 (CAST_SPELL + REPLACE_CARD_MOVE)
  └─ マージテキスト生成: "その呪文を唱えた後、{from}に置くかわりに{to}に置く。"
  ↓
ensure_executable_command()
  └─ map_action() でコマンド形式に変換
  ↓
EngineCompat.ExecuteCommand()
  ├─ C++ 側で実行（未実装時）
  └─ Python フォールバック:
       - compat.py:416 で REPLACE_CARD_MOVE 処理
       - instance_id からカード取得
       - from_zone から削除, to_zone に配置
```

## ✨ 機能動作確認

### UIでの表示
- ✅ コマンドグループ: CARD_MOVE で "置換移動" が表示
- ✅ 個別フィールド表示:
  - Original Destination: 墓地
  - Replacement Destination: 山札の下

### テキスト生成
- ✅ 単独: 「そのカードを墓地に置くかわりに山札の下に置く。」
- ✅ 連鎖: 「その呪文を唱えた後、墓地に置くかわりに山札の下に置く。」

### エンジン実行
- ✅ コマンド変換: 正常に実行可能な形式に変換
- ✅ Python フォールバック: compat.py で処理実装済み
- ✅ 状態変更: インスタンスが正しいゾーン間を移動

## 🚀 デプロイ確認

- ✅ コード変更: `text_generator.py` を修正
- ✅ 日本語化: `ja.json` で REPLACE_CARD_MOVE: "置換移動" 定義済み
- ✅ テスト: 統合テストで全項目パス
- ✅ ドキュメント: このファイルで実装内容を記録

---

**実装日時**: 2026年1月17日  
**ステータス**: ✅ 完了  
**レビュー**: 待機中
