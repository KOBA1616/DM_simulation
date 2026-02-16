# CommandEncoder 設計

目的:
- `Action` ベースのエンコーディングを `Command` ファーストに移行し、ネイティブ（C++ MCTS）と Python トレーニング実装で同一のインデックス空間を共有する。

要件:
- 一意で決定的なインデックススキーマ（`index -> command` とその逆 `command -> index`）を定義する。
- `TOTAL_COMMAND_SIZE`（旧 `TOTAL_ACTION_SIZE` 相当）を確定し、ネイティブと Python 両方で同一値とする。
- 既存の互換層を段階的に削除できるよう、フォールバック・互換 API を維持する。
- パリティテストで Python フォールバック実装とネイティブ実装の出力を比較する。

公開 API（例）:
- `class CommandEncoder:`
  - `@staticmethod def command_to_index(cmd: dict) -> int`
  - `@staticmethod def index_to_command(idx: int) -> dict`
  - `TOTAL_COMMAND_SIZE: int`

インデックススキーマ設計（提案）:
1. 特殊コマンド領域 (reserve for PASS, RESIGN, etc.) — 固定少数の先頭インデックス
   - 例: `0 = PASS`, `1 = RESIGN`。
2. パラメータ化可能コマンド領域 — 別々のセグメントに分割
   - `MANA_CHARGE`: スロット数 S1 → インデックス区間 `[base_mana, base_mana + S1)`
   - `PLAY_FROM_ZONE`: スロット数 S2 → インデックス区間 `[base_play, base_play + S2)`
   - 他のコマンドは同様にセグメントを割り当てる。
3. セグメントは順次割り当て、合計で `TOTAL_COMMAND_SIZE` を算出する。

互換性方針:
- 既存 `Action` を即時削除せず、`dm_ai_module` の互換関数で `Action` をラップし、DeprecationWarning を出す。
- 一度 `CommandEncoder` 設計が確定したら、`scripts/action_to_command_migrate.py` を用いてソースを段階的に置換する（dry-run を必須にする）。

テストと検証:
- 単体: `index_to_command` の逆写像が `command_to_index` と整合する（有限集合で round-trip テスト）。
- パリティ: Python 実装とネイティブ `.pyd` 実装の出力一致（代表的なインデックス集合で確認）。
- 統合: トレーニング + MCTS の end-to-end 行動分布が既存実装から大きく逸脱しないことをベンチで確認。

次のアクション:
- この設計に従って `CommandEncoder` の初期実装（Python）を作成し、`native_prototypes` のネイティブ実装と比較するためのパリティテストを追加する。