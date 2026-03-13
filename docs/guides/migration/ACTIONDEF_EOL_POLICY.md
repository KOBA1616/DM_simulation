# ActionDef EOL Policy

最終更新: 2026-03-13

## 目的

`ActionDef` は旧JSON互換のためだけに残しているレガシー構造です。
新規データは `CommandDef` に統一し、撤去可能な状態へ段階移行します。

## 廃止期限

- EOL 期限: `2026-06-30`
- 期限までの運用:
1. `schema_version` 未指定 or `1`: 旧 `actions` を受理（互換モード）
2. `schema_version >= 2`: `actions` を禁止し、`commands` のみ受理

## 境界仕様（実装済み）

- `src/engine/infrastructure/data/json_loader.cpp`
: `schema_version >= 2` かつ `actions` が存在するカードを拒否
- `src/core/card_json_types.hpp`
: `ActionDef` を legacy と明示し、廃止期限をコメントで固定

## JSON変換手順（旧 -> 新）

1. カードJSONに `schema_version: 2` を追加
2. 各 `effects[*].actions` を `effects[*].commands` に置換
3. `metamorph_abilities[*].actions` も同様に置換
4. 旧 `actions` キーを削除
5. テスト実行:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_json_loader_action_schema_boundary.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_json_loader_conversion.py -q
```

## 撤去判定チェックリスト

- `schema_version >= 2` データで `actions` が使われていない
- 旧互換カードの移行完了（または変換ツール配備）
- `ActionDef` 参照箇所を段階削除して回帰がない
