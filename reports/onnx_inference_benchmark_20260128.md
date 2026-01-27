ONNX Inference Benchmark — 2026-01-28

概要
- モデル: models/duel_transformer_20260123_143241.onnx
- ランタイム: onnxruntime (Python .venv): 1.23.2
- C++ 依存: build uses ONNX Runtime 1.18.0 (build-msvc/_deps/onnxruntime_pkg-src/VERSION_NUMBER)

測定方法
- スクリプト: scripts/bench_onnx_inference.py
- シーケンス長: 200
- Warmup: 10, Iter: 100
- CPU 実行（CPUExecutionProvider）で threads = [1,2,4], batch sizes = [1,2,4,8,16,32]

結果（平均バッチレイテンシ / samples/s）
- threads=1
  - batch=1: avg_batch_ms=66.25 ms, samples/s=15.1
  - batch=2: avg_batch_ms=134.14 ms, samples/s=14.9
  - batch=4: avg_batch_ms=274.61 ms, samples/s=14.6
  - batch=8: avg_batch_ms=718.38 ms, samples/s=11.1
  - batch=16: avg_batch_ms=1738.44 ms, samples/s=9.2
  - batch=32: avg_batch_ms=1793.80 ms, samples/s=17.8

- threads=2
  - batch=1: avg_batch_ms=21.19 ms, samples/s=47.2
  - batch=2: avg_batch_ms=45.65 ms, samples/s=43.8
  - batch=4: avg_batch_ms=92.40 ms, samples/s=43.3
  - batch=8: avg_batch_ms=197.77 ms, samples/s=40.5
  - batch=16: avg_batch_ms=439.02 ms, samples/s=36.4
  - batch=32: avg_batch_ms=912.62 ms, samples/s=35.1

- threads=4
  - batch=1: avg_batch_ms=18.37 ms, samples/s=54.4
  - batch=2: avg_batch_ms=38.23 ms, samples/s=52.3
  - batch=4: avg_batch_ms=82.59 ms, samples/s=48.4
  - batch=8: avg_batch_ms=195.08 ms, samples/s=41.0
  - batch=16: avg_batch_ms=441.07 ms, samples/s=36.3
  - batch=32: avg_batch_ms=938.83 ms, samples/s=34.1

考察と推奨
- 小バッチ（1〜2）を使い、`ORT_INTRA_THREADS` をコア数に合わせて増やすとスループットが最大化される傾向。
- 推奨初期設定（エンジン本番）:
  - `ORT_INTRA_THREADS` = CPU コア数（例: 4）
  - `ORT_INTER_THREADS` = 1〜2
  - MCTS の `batch_size_` を `1` または `2` に設定（デフォルトは 1 か 2 を推奨）

補足
- C++ ビルド側の ONNX Runtime バージョンは 1.18.0 ですが、Python 環境の `onnxruntime` が 1.23.2 でした。CI では `onnxruntime>=1.18.0` に固定してください。

ファイル
- スクリプト: scripts/bench_onnx_inference.py
- 本レポート: reports/onnx_inference_benchmark_20260128.md
