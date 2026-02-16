# Migration

移行ガイドを格納します。

- [MIGRATION_GUIDE_ACTION_TO_COMMAND.md](MIGRATION_GUIDE_ACTION_TO_COMMAND.md)

要約: `MIGRATION_GUIDE_ACTION_TO_COMMAND.md` はレガシーの Action 辞書形式から標準化された GameCommand 構造への段階的移行を説明します。移行の背景、フェーズ（Phase 1〜4）、リファクタリング内容（`action_to_command.py`, `compat_wrappers.py`, `command_builders.py` など）、テスト結果と推奨手順が記載されています。

主な見出し: 概要 / 移行の背景と目的 / 実装内容 / テスト結果 / 段階的移行パス / 使用ガイド

移行作業を行う際は、各フェーズのチェックリストに従って段階的に適用してください。
