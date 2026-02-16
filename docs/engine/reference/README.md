# Reference

リファレンス文書をまとめたフォルダです。

- [SPELL_EXECUTION_TIMING_QA.md](SPELL_EXECUTION_TIMING_QA.md)
- [SPELL_REPLACEMENT_QUICK_REF.md](SPELL_REPLACEMENT_QUICK_REF.md)
- [SPELL_ZONE_FLOW_AND_REPLACEMENT.md](SPELL_ZONE_FLOW_AND_REPLACEMENT.md)

各ファイルの短い要約:

- SPELL_EXECUTION_TIMING_QA.md — 呪文効果の実行タイミングに関するQ&A。結論として「効果はすべてSTACK上で実行され、その後ゾーン移動が行われる」を詳細に示す。
	- 主な見出し: Q&A / 詳細説明 / コードによる証明 / 重要な結論
- SPELL_REPLACEMENT_QUICK_REF.md — 置換効果（REPLACE_CARD_MOVE）に関するクイックリファレンス。通常のフローと置換効果ありのフローを比較している。
	- 主な見出し: ゾーン経路比較 / 重要な違い / 実装の核心部分 / GUIでの設定手順
- SPELL_ZONE_FLOW_AND_REPLACEMENT.md — 呪文のゾーン経路と置換効果の実装ガイド（詳細な実装手順とタイムラインを含む）。
	- 主な見出し: 通常の呪文解決フロー / 置換効果の処理 / 実装方法 / テスト / 使用例

参照用の短い抜粋やコードポイントをここに追加しておくと便利です。
