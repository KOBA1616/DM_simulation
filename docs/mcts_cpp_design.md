# MCTS C++化設計（概要）

目的
- `MCTS::search` を完全に C++ 実装に移行し、MCTS ループ内で Python コールバック（GIL）を呼ばない。
- 推論パスはゼロコピー（`py::array_t` -> 生ポインタ）を使い、メモリコピーを排除する。
- 起動時にネイティブモデルが必ずロードされていることを前提とし、実行時の Python フォールバックを禁止または最小化する。

設計要件
- バッチ評価インタフェース（同期/非同期）を C++ 側で提供する。ホットパスはゼロコピー。
- スレッド安全なバッチキューとスレッドプール（将来的に非同期 infer を返却できる future）。
- `NativeInferenceManager::infer_flat_ptr(const float*, size_t, int, int)` を利用する。
- Python 側へのコールは起動時チェックのみ許可。実行中に Python コールに落ちる場合はログとエラーで明確にする。
- float32 を前提とし、double を避ける。

主要コンポーネント（ファイル）
- `src/ai/mcts/mcts.hpp` / `mcts.cpp` — 既存実装の移行先。`MCTS::search` をバッチ評価に依存する形に書き換え。
- `src/ai/mcts/mcts_evaluator.hpp` / `mcts_evaluator.cpp` — 新規バッチ評価器。
  - クラス `BatchEvaluator`
    - public:
      - `BatchEvaluator(size_t max_batch_size, size_t num_workers = std::thread::hardware_concurrency());`
      - `std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate(const std::vector<std::shared_ptr<GameState>>& states);` // 同期呼び出し（現段階）
      - 将来の拡張: `std::future<...> evaluate_async(...)`
    - 内部:
      - バッファリング用 preallocated buffer、条件変数、ワーカー群
      - 呼び出し時は `states` から `py::array_t` と互換な flat 配列を作らず、代わりに C++ エンコーダを用いゼロコピーで `infer_flat_ptr` を呼ぶ（既に状態が flat float の場合は直接参照）。
- `src/bindings/python_batch_inference.cpp` — 既に起動時フォールバック化済み。バッチ評価器の公開 API を追加して Python 側で `BatchEvaluator` をオプションで参照できるようにする（互換性確保のため）。
- `src/ai/inference/native_inference.hpp/.cpp` — 既に `infer_flat_ptr` を追加済み。必要に応じて int8/float16 の受け口も拡張。

API シグネチャ提案（抜粋）
- C++:
  - class BatchEvaluator {
    - `BatchEvaluator(int batch_size, int num_workers = 0);`
    - `std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate(const std::vector<std::shared_ptr<GameState>>& states);`
  - }
- pybind11 バインディング:
  - `m.def("create_batch_evaluator", &create_batch_evaluator);` // Python での作成・指定が可能

MCTS 側の呼び方（要点）
- `MCTS::search` の中では評価が必要なノードを集め、`BatchEvaluator::evaluate(batch_states)` を呼ぶ。
- 戻り値は `policies: vector<vector<float>>`（各ノード毎の action ロジット/確率）と `values: vector<float>`。
- `expand_node` はこれらを受け取ってノードの prior/value を設定する。

スレッドモデル
- MCTS::search は外部から複数スレッドで呼ばれる可能性を考慮し、内部同期は `MCTSNode` 単位の mutex またはトランスポジションテーブルでの atomic 操作を用いる。
- BatchEvaluator は内部にワーカースレッドを持ち、推論ライブラリ呼び出しはワーカー上で行われる（GIL を取らない）。

エラーとフォールバック方針
- 起動時に `NativeInferenceManager::has_model()` を確認する。false の場合、起動時にユーザへエラー（または `-AllowFallback` が指定されていれば警告）を出す。実行時に native が失敗した場合は例外を投げて処理を停止する（ループ中での Python フォールバックは行わない）。

互換性戦略
- 既存の Python `dm_toolkit.ai.agent.mcts` は当初は shim を使い `dm_ai_module.MCTS` が利用できない環境では従来 Python MCTS を使う。CI/デフォルトではネイティブ経路を必須とする運用を推奨。

検証/ベンチ項目
- GIL 待ち時間（py-spy / Windows profiler）
- memcpy 回数と総帯域（簡易メモリプロファイラか Windows のツール）
- レイテンシとスループット（`bench_onnx_inference.py` と MCTS 単体ベンチ）

短期ロードマップ（推奨順）
1. この設計ドキュメント合意
2. `BatchEvaluator` の最小同期実装（`evaluate()` は同期で動作）を追加し、ユニットテストを作成
3. `MCTS::search` を evaluator 呼び出しに差し替え、単体テストで動作確認
4. バインディングを追加して Python から新 `MCTS` を呼べるようにする
5. 非同期/スレッドプール最適化、さらにメモリプールと float32 徹底
6. プロファイリング実行と CI 組込み

次のアクション提案
- すぐに `BatchEvaluator` の同期版を実装します（推奨）。
- これを希望する場合、続けて実装ファイルを作成します。

---
設計に修正や追加希望があれば指定してください。