# DM_TOOLKIT_FORCE_DUMMY_QT CI Policy

目的: Qt 初期化順序に起因する順序依存エラーを抑えつつ、GUI 回帰検知能力を維持する。

## 方針

- `headless` ジョブ: `DM_TOOLKIT_FORCE_DUMMY_QT=1`
- `gui` ジョブ: `DM_TOOLKIT_FORCE_DUMMY_QT=0`

実装先: `.github/workflows/ci.yml`

## 理由

- headless 実行では、`QApplication` の初期化順序がテスト順に依存して不安定化することがある。
- ダミー Qt を強制すると、ヘッドレステストの安定性を優先できる。
- 一方で GUI 回帰は実 Qt でしか検出できないため、GUI ジョブでは強制しない。

## 運用ルール

- 新規 CI ジョブを追加する場合:
  - GUI テストを含まないジョブは `DM_TOOLKIT_FORCE_DUMMY_QT=1` を検討する。
  - GUI ウィジェット挙動を検証するジョブは `DM_TOOLKIT_FORCE_DUMMY_QT=0` を明示する。
- ローカル再現時:
  - headless 再現: `set DM_TOOLKIT_FORCE_DUMMY_QT=1`
  - GUI 再現: `set DM_TOOLKIT_FORCE_DUMMY_QT=0`
