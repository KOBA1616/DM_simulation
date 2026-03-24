# ONNX Runtime Version Policy

最終更新: 2026-03-13

## 目的

ネイティブ推論経路で発生しやすい DLL/API ミスマッチを防ぐため、
Python 実行時と C++ ビルド時の ONNX Runtime バージョンを単一点で固定する。

## 固定バージョン

- 正式固定: `1.20.1`
- 変更対象:
1. `requirements.txt`
2. `pyproject.toml`
3. `CMakeLists.txt` の ORT ダウンロード URL

## 運用ルール

1. ORT バージョンを上げるときは、上記 3 か所を同一コミットで更新する。
2. 更新後に整合テストを実行する。
3. ネイティブ推論スモーク (`tests/test_transformer_inference_native.py`) を実行する。

## 検証コマンド

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_onnxruntime_version_alignment.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_transformer_inference_native.py -q
```

## 不整合時の典型症状

- `native_load_onnx` で API mismatch を検知して終了コード `6` を返す
- `onnxruntime*.dll` のロード失敗
- テストが環境依存で通ったり失敗したりする

## トラブルシューティング

1. `scripts/check_ort_runtime_mismatch.py` を実行して、`reports/ort_runtime_check.txt` を確認する。
2. `site-packages/onnxruntime` と CMake が取得した ORT バージョンが一致しているか確認する。
3. ワークスペース直下に古い `onnxruntime*.dll` が残っている場合は整理して再実行する。
