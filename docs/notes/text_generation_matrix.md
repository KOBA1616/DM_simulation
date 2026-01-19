# テキスト生成マトリクス

以下は `dm_toolkit.gui.editor.text_generator.CardTextGenerator` の `ACTION_MAP` と
`_format_action` / `_format_command` に基づく、アクション型 → 生成されるテキスト関係の網羅表です。

| アクション型 | 生成テンプレート（原文） | 条件 / 特記事項 | 主に使用するパラメータ |
|---|---:|---|---|
| DRAW_CARD | カードを{value1}枚引く。 | 単純描画テンプレ。`TRANSITION`(DECK→HAND) の特殊扱いでも同様に「カードをN枚引く。」になる。 | `value1`, `input_value_key` |
| ADD_MANA | 自分の山札の上から{value1}枚をマナゾーンに置く。 | `TRANSITION`→`to_zone==MANA_ZONE` の特殊処理でも同様テキストに変換される。 | `value1`, `amount`, `target_filter` |
| DESTROY | {target}を{value1}{unit}破壊する。 | `value1==0` や入力リンクで「すべて」や「その数」に置換。 `TRANSITION` (BATTLE_ZONE→GRAVEYARD) にもマッピングあり。 | `value1`, `filter`, `scope` |
| TAP | {target}を{value1}{unit}選び、タップする。 | `value1==0` →「すべてタップする」。`mutation_kind==TAP` の `MUTATE` でも同様表現。 | `value1`, `filter`, `mutation_kind` |
| UNTAP | {target}を{value1}{unit}選び、アンタップする。 | 同上（選択数0→すべて）。 | `value1`, `filter` |
| RETURN_TO_HAND | {target}を{value1}{unit}選び、手札に戻す。 | `TRANSITION` で from= BATTLE_ZONE→HAND 等にも対応。 | `value1`, `destination_zone`, `from_zone` |
| SEND_TO_MANA | {target}を{value1}{unit}選び、マナゾーンに置く。 | `MANA_CHARGE` コマンドは scope により `ADD_MANA` / `SEND_TO_MANA` に分岐。 | `value1`, `destination_zone`, `target_filter` |
| MODIFY_POWER | {target}のパワーを{value1}する。 | 符号を付けて表示（例: +500）。`MUTATE` の POWER_MOD もここへマップする。 | `value1`, `mutation_kind` |
| BREAK_SHIELD | 相手のシールドを{value1}つブレイクする。 | - | `value1` |
| LOOK_AND_ADD | 自分の山札の上から{value1}枚を見る。その中から{value2}枚を手札に加え、残りを{zone}に置く。 | - | `value1`,`value2`,`zone` |
| SEARCH_DECK / SEARCH_DECK_BOTTOM | 山札検索系のテンプレ | `filter` に依存し、戻し先 `{zone}` を用いる | `filter`, `zone` |
| MEKRAID | メクレイド{value1}（...） | 固有長文テンプレ | `value1` |
| DISCARD | 手札を{value1}枚捨てる。 | `scope==NONE` のとき `_resolve_target` が手札を指す | `value1`, `scope` |
| PLAY_FROM_ZONE | {source_zone}からコスト{value1}以下の{target}をプレイしてもよい。 | `filter.types` によって「唱える/召喚する/プレイする」に動的決定 | `source_zone`, `filter.types`, `value1` |
| COUNT_CARDS | {filter}の数を数える。 | `target_str` が空だと短縮表現で括弧表記 | `filter` |
| GET_GAME_STAT | （{str_val}を参照） | 多くは空文字返却の特殊処理あり | `str_val` |
| REVEAL_CARDS | 山札の上から{value1}枚を表向きにする。 | - | `value1` |
| SHUFFLE_DECK | 山札をシャッフルする。 | - | なし |
| ADD_SHIELD | 山札の上から{value1}枚をシールド化する。 | - | `value1` |
| SEND_SHIELD_TO_GRAVE | 相手のシールドを{value1}つ選び、墓地に置く。 | - | `value1` |
| SEND_TO_DECK_BOTTOM | {target}を{value1}枚、山札の下に置く。 | - | `value1`, `destination_zone` |
| CAST_SPELL | {target}をコストを支払わずに唱える。 | 主に `PLAY_FROM_ZONE` と区別される用途 | `target` |
| PUT_CREATURE | {target}をバトルゾーンに出す。 | - | `target` |
| GRANT_KEYWORD | {target}に「{str_val}」を与える。 | `ADD_KEYWORD` コマンドは `GRANT_KEYWORD` にマップ。`MUTATE` の ADD_KEYWORD と同様表現。 | `str_val`, `filter` |
| MOVE_CARD | {target}を{zone}に置く。 | `destination_zone` による分岐（HAND/MANA/GRAVE/DECK_BOTTOM） | `destination_zone`, `value1` |
| COST_REFERENCE | <空文字 or 専用文言> | `str_val` により G_ZERO / HYPER_ENERGY / SYM_* 等の専用文言へ変換 | `str_val`, `condition` |
| SUMMON_TOKEN | 「{str_val}」を{value1}体出す。 | - | `str_val`,`value1` |
| RESET_INSTANCE | {target}の状態をリセットする（アンタップ等）。 | - | `target` |
| REGISTER_DELAYED_EFFECT | 「{str_val}」の効果を{value1}ターン登録する。 | - | `str_val`,`value1` |
| FRIEND_BURST | {str_val}のフレンド・バースト | 固有書式（特殊効果説明付き） | `str_val` |
| MOVE_TO_UNDER_CARD | {target}を{value1}{unit}選び、カードの下に置く。 | - | `value1` |
| SELECT_NUMBER / DECLARE_NUMBER | 数字を選ぶ系テンプレ | `value1`value2` 範囲指定で生成 | `value1`,`value2` |
| COST_REDUCTION | {target}のコストを{value1}少なくする。 | `_format_condition` が前置される場合あり。 | `value1`,`condition` |
| LOOK_TO_BUFFER / SELECT_FROM_BUFFER / PLAY_FROM_BUFFER / MOVE_BUFFER_TO_ZONE | バッファ操作系テンプレ | バッファ固有文言 | `value1`,`filter` |
| SELECT_OPTION | 選択肢列挙テキスト（複数行） | `options` により枝分かれを列挙 | `options`,`value1` |
| LOCK_SPELL | 相手は呪文を唱えられない。 | - | なし |
| APPLY_MODIFIER | 効果を付与する。 | `str_val` により「スピードアタッカー」等へ翻訳。 | `str_val`,`value1` |
| TRANSITION | {target}を{from_zone}から{to_zone}へ移動する。 (フォールバック) | 多数の自然言語マッピング有り：
| | | - DECK→HAND -> "カードを{amount}枚引く。"（`target_filter.count` も参照）
| | | - to MANA_ZONE -> "自分の山札の上から{amount}枚をマナゾーンに置く。"
| | | - BATTLE_ZONE→GRAVEYARD -> 破壊表現
| | | - BATTLE_ZONE→HAND/MANA_ZONE -> 手札/マナ表現（選択数0→すべて）
| | | 特記事項: `amount==0` は「すべて」と扱う、`{from_z}`/`{to_z}` は `tr()` で翻訳される。 | `from_zone`,`to_zone`,`amount`,`target_filter` |
| MUTATE | {target}の状態を変更する。 (フォールバック) | `mutation_kind` により TAP/UNTAP/POWER_MOD/ADD_KEYWORD/REMOVE_KEYWORD 等へ分岐し、各々専用テキストを返す。 | `mutation_kind`,`amount`,`str_param` |
| FLOW | 進行制御: {str_param} | `flow_type` が PHASE_CHANGE/TURN_CHANGE 等なら専用文言 | `flow_type`,`value1` |
| QUERY | クエリ発行: {query_mode} | `query_mode` を `tr()` で表示 | `query_mode` |
| ATTACH | {target}を{base_target}の下に重ねる。 | - | `base_target`,`target` |
| GAME_RESULT | ゲームを終了する（{result}）。 | - | `result` |

---

注意事項（共通）:

- `input_value_key` が設定されると、テンプレの数値は「その数」表現に置換される（例: "その数だけ破壊する"）。
- アクションが `optional=True` の場合、文末の活用（「する。」→「してもよい。」等）に変換される。
- 未定義テンプレは `(tr(ATYPE))` のように型名を括弧で返す（フォールバック）。
- `_format_command` はコマンド型をアクション風にプロキシ変換する（`POWER_MOD`→`MODIFY_POWER`、`ADD_KEYWORD`→`GRANT_KEYWORD`、`MANA_CHARGE`→`SEND_TO_MANA/ADD_MANA` 等）。

このファイルは人によるレビュー用の要約です。自動で CSV/JSON 出力が必要なら生成します。ご希望はどちらですか？
