# scripts/

開発用の補助スクリプト置き場です。用途別に次を参照してください。

- 環境セットアップ
  - `init_env.ps1`（venv準備/起動の入口）
  - `setup_build_env.ps1` / `setup_mingw_env.ps1`
  - `setup_clone_windows.ps1`（クローン配布向け: 環境確認→venv→依存導入→ビルド→スモークテスト）
  - `setup_gui_review_windows.ps1`（GUIレビュー専用: PyQt6のみ導入→カードエディタ起動。ネイティブ不要）

- ビルド/テスト/CI
  - `build.ps1`（ローカルビルド）
  - `run_ci_local.ps1`（CI相当をローカルで実行してログ保存）
  - `run_build_with_vs.cmd`

- GUI
  - `run_gui.ps1`
  - `run_gui_with_real_pyqt.ps1`
  - `run_gui_review.ps1`（GUIレビュー用途: カードエディタを素早く起動。事前に setup_gui_review_windows.ps1 実行）

- メンテ/クリーン
  - `clean_workspace.ps1`
  - `clean_msvc.ps1`

- GitHub/PR作業
  - `gh_setup_and_pr.ps1`
  - `create_pr_and_watch.ps1`

- Python補助スクリプト
  - `python/` 配下

ドキュメント類は原則 `docs/` に置きます（例: テキスト生成マトリクスは `docs/notes/text_generation_matrix.md`）。
