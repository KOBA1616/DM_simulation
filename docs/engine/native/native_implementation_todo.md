# Native Implementation TODO

このドキュメントは、現行リポジトリでPythonフォールバック（shim）を入れてテストを通した上で、恒久的にC++ネイティブ側で実装すべきシンボル・機能を優先度付きで整理したものです。

作成日: 2026-01-25

---

## 要約（高優先度）
- 目的: 性能と一貫性を取り戻すため、短期的に Python フォールバックに依存しないようにする。
- 期待効果: CI 上での安定性向上、速度改善、バグの根本解消。

---

## 優先度: P0（即対応推奨）
1. DataCollector (C++):
   - 理由: 大量データ収集・バッチ推論でパフォーマンスに直結するため。
   - 期待インターフェース: `collect_data_batch_heuristic(batch_size, include_history, include_features)` 等。
   - 参照箇所: `training/*`, `tests/test_game_flow_minimal.py`。
   - 見積: 2-3日（既存 C++ コードベース参照で実装）

2. ParallelRunner / 推論ブリッジ (C++):
   - 理由: ネイティブ並列推論のライフサイクルとバッチ登録APIが重要。
   - 期待インターフェース: `ParallelRunner(card_db, sims, batch_size)`, `play_games(...)`, `register_batch_inference_numpy` 等。
   - 参照箇所: `dm_ai_module` 呼び出し箇所、`tests/test_native_inference_bridge.py`。
   - 見積: 3-5日

3. CardDatabase 高速アクセスメソッド (C++):
   - 理由: カード定義参照が頻繁で、ネイティブ側で最適化すべき。
   - API: `get_card(id)`, イテレータ、ローディングの一貫性。
   - 参照: `JsonLoader` の代替・補完。
   - 見積: 1-2日

---

## 優先度: P1（重要）
1. EffectResolver / Effect 系 (C++):
   - 理由: 効果解決ロジックはゲームルールの核心。正確な仕様で実装する必要あり。
   - 参照: `EffectDef`, `EffectActionType`, `tests/test_spell_and_stack.py` 等
   - 見積: 3-7日（仕様の精査が必要）

2. Command / Flow 系 (ネイティブの Command, FlowCommand 実装):
   - 理由: 攻撃フローやイベント伝播で低レイテンシが求められる。
   - 参照: `FlowCommand`, `FlowType`, `CommandDef`, `CommandSystem` のネイティブ実装
   - 見積: 2-4日

3. TokenConverter / TensorConverter (C++):
   - 理由: モデル入力用トークナイズ／テンソル変換の効率化。
   - 参照: `training/*`, `tests/test_native_inference_bridge.py`
   - 見積: 1-3日

---

## 優先度: P2（中長期）
- MutateCommand / MutationType（もし複雑な副作用があるならネイティブに移行）
- IntentGenerator / CommandEncoder（ルールベースの高速実装が必要な場合）
- DeckEvolution, DeckInference（高度機能）

---

## 実装ガイドライン（提案）
- API 安定化: Python shim と同じ名前・引数で提供し、既存の Python 呼び出しが透過的にネイティブを使えるようにする。
- テスト駆動: ネイティブ実装を進める際、まず Python 側のテスト（`tests/`）を動かしながら進める。
- ドキュメント: 変更点は `docs/` に記載し、C++ 側のヘッダーで公開 API を明記。

---

## 次のアクション（実務）
1. ここに列挙した P0/P1 のシンボルについて、担当（C++ dev）と日程を決める。
2. `dm_ai_module.py` の shim を短期的に維持しつつ、ネイティブ側実装をマージしていく。
3. 完了後、`scripts/check_native_symbols.py` を再実行して残差を確認する。

---

必要であれば、このドキュメントをさらに詳細化し、各シンボルの期待ヘッダ/シグネチャを生成します。どのシンボルからシグネチャ化しますか？（例: `DataCollector`／`ParallelRunner`）
