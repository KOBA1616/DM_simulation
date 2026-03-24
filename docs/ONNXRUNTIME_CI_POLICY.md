# ONNX Runtime CI Policy

目的: ONNX Runtime (`onnxruntime`) のバージョン差分が原因で CI やローカル検証が不安定になることを防ぐ。

方針:
- CI 環境では ONNX Runtime を明示的にピン (`1.20.1`) してインストール/バンドルする（参照: `CMakeLists.txt`, `.github/workflows/ci-pin-onnxruntime.yml`）。
- Python 側の依存関係にも同一バージョンを pin して整合性を保つ（`requirements.txt` / `pyproject.toml`）。
- テストランタイムでインストール済みの `onnxruntime` のバージョンがピンと異なる場合、該当テストは `xfail` として扱う。これにより、CI 上での誤検出を避けつつ不整合は監視可能にする（実装例: `tests/test_onnxruntime_version_alignment.py`）。

運用手順:
1. バージョンを更新する場合は、必ず次のファイルを同一コミットで更新すること:
   - `requirements.txt` または `pyproject.toml`
   - `CMakeLists.txt` の ONNX Runtime のダウンロード URL
   - `.github/workflows/ci-pin-onnxruntime.yml`（CIで明示的にインストール/バンドルする場合）
2. 更新後、`tests/test_onnxruntime_version_alignment.py` をローカルで実行して整合性を確認する。
3. 可能ならば、新バージョンでユニット/統合テストをフル実行して互換性を検証する。

理由:
- ONNX Runtime はプラットフォーム別バイナリの違いとバージョン依存が強く、ランタイム差分が原因でCI失敗が発生しやすい。
- ピンと `xfail` 方針の組合せにより、CI上は再現可能な基準（ピン）を守りつつ、実行環境差分は明示的に管理できる。

メンテナンス:
- ONNX Runtime のピンは年に1回以上レビューし、必要ならばまとめてバージョンアップ作業を計画する。
