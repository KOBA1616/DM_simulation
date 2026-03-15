# Migration PR Draft: schema_config -> dm_toolkit.consts

目的
- `dm_toolkit/gui/editor/schema_config.py` 内の選択肢定義 (`DURATION_OPTIONS` など) を
  `dm_toolkit.consts` の定義へ完全に委譲し、定義の単一化を行う。

背景
- 現在 `schema_config.py` は一部を `dm_toolkit.consts` から参照しているが、TODO コメントが残る。
- 既に `DURATION_OPTIONS = DURATION_TYPES` のような委譲は行われているが、PR では残件の整理と説明、CI 通過を目指す。

変更内容（概要）
1. `schema_config.py` のトップレベル定数（`DURATION_OPTIONS`, `TARGET_SCOPES` 等）を
   `dm_toolkit.consts` のエイリアスに統一する（既存のコード互換を維持）。
2. ファイル末尾に移行注記を残し、ドキュメント `docs/migration_schema_config_to_consts_PR.md` を追加する。
3. 既存ユニットテスト（`python/tests/unit/test_schema_config_sync.py` 等）を更新/追加して
   `schema_config` と `dm_toolkit.consts` の整合性を CI で検証する。

差分サンプル
- 変更前（抜粋）: `DURATION_OPTIONS = ['TURN', 'UNTIL_END', ...]`
- 変更後（抜粋）: `from dm_toolkit.consts import DURATION_TYPES as DURATION_OPTIONS`

テスト手順
1. ローカルでユニットテストを実行: `pytest python/tests/unit/test_schema_config_sync.py -q`
2. フルテスト: `pytest -q`（CI が許すなら）

PR コメント（説明）
- この PR は定数の正規化を目的としています。`dm_toolkit.consts` を SSOT とし、
  `schema_config.py` は UI スキーマ定義に専念します。互換性は保たれ、既存の機能に
  影響を与えないことを確認済みです（付属テストあり）。

注意点
- 重大な変更は避け、まずはエイリアス化で差分を小さく抑える。将来的に `consts` へ
  より多くを移す際は段階的に行う。
