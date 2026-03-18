# ONNX Runtime (ORT) Version Policy

目的: CI とローカル開発で ONNX/ORT に起因するテスト不一致を防ぐための運用方針を定義します。

短い要約:
- CI と開発環境で使用する `onnxruntime` のバージョンを明示的に管理し、CMake のフェッチ設定と Python の `requirements` を一致させます。
- テストに依存する ONNX IR/opset 要件は、リポジトリ内の参考値（`docs/ORT_VERSION_POLICY.md`）へ記載し、onnx モデル生成スクリプトは互換性を意識して opset を明示します。

推奨手順（運用）:
1. CI のベースイメージ/ジョブで使う ORT バージョンを決定する。
   - 例: `onnxruntime==1.15.1`（一例。実際に CI でテストして決定してください）
2. `requirements.txt` / `requirements-dev.txt` に Python pin を追加する。
   - 例: `onnxruntime==1.15.1`
3. CMake の外部フェッチ（もしあれば）で取得している ORT のバージョンと一致させる。
   - `cmake/` または `CMakeLists.txt` に記載のフェッチ仕様を確認し、ピンを合わせる。
4. テストの期待値について:
   - 互換性差による挙動差が見られるテストは `xfail` を使って明示的に扱う（ただし恒久的な xfail は避け、根本対応を追う）。
5. モデル生成スクリプトは互換性を持つ opset/IR を使う。
   - テスト用 runner では `opset_imports` と `model.ir_version` を明示して互換性を確保する（本リポジトリ内の `scripts/run_native_onnx_loader.py` を参照）。

次の実務タスク（提案）:
- (A) CI ジョブの ORT バージョンを決め、`requirements*` へ pin を適用（PR を作成）。
- (B) `tests/test_onnxruntime_version_alignment.py` の期待値を更新して CI ポリシーに合わせる（xfail を解除できるか確認）。

参考: 本ドキュメントは運用ポリシーの草案です。実際のバージョン選定は CI 上での安定性確認を経て確定してください。
