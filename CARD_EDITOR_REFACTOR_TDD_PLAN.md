
# カードエディタ改善・統合・データ構造再設計 実装計画

最終更新: 2026-03-18（test_results.log 反映）

対象: `dm_toolkit/gui/editor/` 一式
目的: 未完了タスクに集中し、小さな TDD サイクルで安全に改善を進める


## 0. 現状

最終確認日: 2026-03-18
最終更新日: 2026-03-18（ログ欠損を含むフルテスト再トリアージ）

### 最新テスト観測（2026-03-18）
- フルテストを実行し、`test_results.log` を保存しました（出力ファイルあり）。
- 実行サマリ（pytest 出力）:
  - collected 364 items
  - 363 passed, 1 xfailed
  - 実行時間: 55.49s
- 備考: 全体が GREEN（ただし 1 件 xfail は既知の ORT バージョン差分で意図的に xfail）。

### 実装済み（概要）
- 多数の修正・テスト追加・運用スクリプトを完了しました。詳細な変更履歴はアーカイブに移動しています。
  - 参照: [archive/](archive/) 内の該当ファイル（例: [archive/implementation_2026-03-17.md](archive/implementation_2026-03-17.md#L1)）を参照してください。
  - 本ドキュメントは今後、未完了タスクに集中するため、完了済みの詳細は原則アーカイブへ移動します。


## 1. 未完了バックログ（2026-03-18 時点）

### P0（今すぐ）

1. フルテスト結果の確定取得
  - 状態: 完了（`test_results.log` にフル出力を保存、最終サマリ取得済み）
  - 次アクション:
    - （対応済）`scripts/run_native_onnx_loader.py` を追加して子プロセスによるネイティブ ONNX ロード検証を実装。
    - 本日追加されたスクリプトは ONNX モデルを生成（pure-onnx、opset=11、IR=10）して Python ORT でロード確認後、`dm_ai_module` のネイティブローダ/`NeuralEvaluator` を呼び出して `INFER_OK` を出力することでテストを通過させます。
  - 完了条件:
    - 失敗一覧と最終サマリを確定できる（完了）
    - 本資料の「最新テスト観測」を確定値へ更新済み

2. `tests/test_card1_hand_quality.py` 4失敗の再現性判定
  - 状態: 継続中（再現試行を追加、依然としてクラッシュ未再現）
  - 最新実施（2026-03-18）:
    - `scripts/repeat_native_loader.py --count 50` を実行: **50/50 PASS**（`INFER_OK`）
    - `scripts/run_tests_shuffled.py -n 5 --log-dir logs/shuffled3` を実行: **5/5 PASS**
    - 追加採取手順を整備: `docs/guides/windows_native_crash_collection.md`
    - `scripts/run_tests_shuffled.py` を拡張し、`summary.json`（反復結果・失敗ファイル・実行順）を出力するようにしました。
    - 実行確認: `scripts/run_tests_shuffled.py -n 1 --log-dir logs/shuffled4` を実行し、`logs/shuffled4/summary.json` 生成を確認しました。
  - 完了内容の削除方針:
    - 本項目の過去の完了済み履歴は削除し、最新結果と次アクションのみ保持する。
  - 次アクション:
      - 実施: 再現トリアージ用ハーネスを追加しました: `scripts/repeat_card1_hand_quality.py`。
        - 目的: 特定テスト `tests/test_card1_hand_quality.py::test_specific_old_card_goes_to_deck_bottom` を反復実行し、ログと `summary.json` を収集して再現性を確認する。
        - 実行コマンド例:
          ```powershell
          python scripts/repeat_card1_hand_quality.py -n 20 --log-dir logs/repro_card1
          ```
        - 出力: `logs/repro_card1/run_{i}.log`, `logs/repro_card1/summary.json`, 失敗時は `logs/repro_card1/last_failure.log` を生成します。
      - 初回実行結果（ローカル、2026-03-18）: `-n 5` を実行 → `Runs: 5, Failures: 5`（すべて失敗）。原因はハーネスのデフォルト nodeid 指定がクラス内テストを正しく参照していなかったためで、実装を修正しました。
      - 修正後の結果: `-n 3` を実行 → `Runs: 3, Failures: 0`（`logs/repro_card1/summary.json` を参照）。
      - 次アクション:
            - ネイティブを最新化した状態で同試行（`repeat_native_loader` / `run_tests_shuffled`）を再実行する。
            - コード変更: `TransitionCommand::execute` の欠損インスタンス診断を強化しました（詳細スナップショットおよび機械可読ダンプを `logs/transition_snapshots/` に出力）。
              - 注意: 変更は C++ 側のソースに加えたため、`dm_ai_module` のネイティブ再ビルドが必要です（CI runner または Visual Studio/MSVC 環境）。
            - 再現時は `dumps/`、`logs/native_repeat/`、`logs/shuffled3/` をセットで保存する。
                - 解析: 追加したスナップショットを素早く集約するためのパーサを `scripts/parse_transition_snapshots.py` として追加しました。
                  - 使い方:
                    ```powershell
                    python scripts/parse_transition_snapshots.py -i logs/transition_snapshots -o logs/transition_snapshots/summary.json
                    ```
                  - 出力: `logs/transition_snapshots/summary.json`（ファイル一覧＋集計）
                - ネイティブ再現支援スクリプト: `scripts/run_native_repro_windows.ps1` を追加しました（Windows 用）。
                  - 概要: 任意にネイティブをビルドし、`scripts/repeat_native_loader.py` を proc-dump でラップして反復実行します。DryRun オプションで実行コマンドのプレビューができます。
                  - 使い方（DryRun で確認）:
                    ```powershell
                    powershell -File scripts/run_native_repro_windows.ps1 -DryRun -RunCount 100 -LogDir logs/native_repro
                    ```
                  - 注意: 実行には CMake / MSVC（または互換ツールチェイン）と ProcDump が必要です。CI またはビルド可能マシンでの実行を推奨します。

### P1（短期）

3. ORT バージョン運用方針の固定
  - 状態: 完了（2026-03-18）
  - 実施内容（今回）:
    - `tests/test_onnxruntime_version_alignment.py` のランタイム検証を厳密化し、バージョン不一致時の `xfail` を廃止しました（不一致は即 FAIL）。
    - 再発防止として、C++ FetchContent と Python 環境のズレを見逃さない方針をテストコードコメントに明記しました。
    - `scripts/check_ort_pin.py` を実行し、`Pinned: 1.20.1  Installed: 1.20.1` を確認しました。
    - `tests/test_onnxruntime_version_alignment.py` を実行し、`3 passed` を確認しました。
  - 補足:
    - 検証途中で `conftest.py` の `ensure_qt_app`（autouse generator fixture）が Qt 非導入時に `yield` せず `ValueError` になる不具合を発見し修正しました。これにより本タスクの検証テストが安定して実行可能になりました。
  - 完了条件:
    - CI/ローカルで利用する pin とランタイムの一致を厳密テストで検証できる状態（達成）。

4. SELECT 系回帰監視の運用化
  - 状態: 完了（2026-03-18）
  - 次アクション:
    - 状態: 完了（監視スクリプトを追加し、短期モニタリングを実行）
    - 実施内容:
      - 監視スクリプトを追加: `scripts/run_select_regression.py`（`tests/test_transition_reproducer.py` と `tests/test_card1_hand_quality.py` を反復実行しログを `logs/select_regression/` に保存）
      - 3回連続実行で PASS を確認（ログ: `logs/select_regression/select_run_{1..3}.log`）。
    - 次アクション:
      - このスクリプトを CI のナイトリーモニタリングジョブに追加するか、開発者が手動で実行して新しい順序依存を検出する運用を推奨します。
    - 実施内容（今回）:
      - 監視手順を `docs/guides/select_regression_runbook.md` に集約しました（実行コマンド、判定基準、失敗時手順を明記）。
      - `scripts/run_select_regression.py` を拡張し、機械可読な `summary.json`（`failed_test`/`log_path` 含む）を出力するようにしました。
      - 実行確認: `-n 1` で監視を実行し、`logs/select_regression/summary.json` と `logs/select_regression/select_run_1.log` の生成を確認しました。
    - 完了条件:
      - 監視スクリプトがリポジトリに存在し、短期モニタリングで問題が発生しないことを確認済み（本フェーズは完了）。
  - 完了条件:
    - 監視対象と失敗時手順が手順書として1箇所にまとまっている（達成）


## 2. 実行ルール

- 1回の実装は 1タスク・1症状・1〜3ファイル変更を原則とする
- 必ず `RED -> GREEN -> REFACTOR` で進める
- 実装後は関係する最小テストを優先実行し、必要に応じてフルテストを実行する
- エラー修正時は再発防止コメントを該当実装へ追加する

