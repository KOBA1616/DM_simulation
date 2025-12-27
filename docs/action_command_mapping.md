# Action → Command マッピング (Schema Definition)

このドキュメントは、システム内で利用される Command 辞書の正規スキーマ（Schema）と、旧 Action/EffectActionType からの変換ルールを定義します。

## 1. Command Dictionary Schema

全ての Command 辞書は以下の共通フィールドを持ちます。

| Key | Type | Description |
| :-- | :-- | :-- |
| `type` | String | Command種別（必須）。例: `TRANSITION`, `MUTATE`, `QUERY` |
| `uid` | String | ユニークID（必須）。通常は UUID 文字列。 |
| `legacy_warning` | Boolean | 変換不完全な場合の警告フラグ（`True` の場合、変換要確認）。 |
| `input_value_key` | String | (Optional) 直前の処理結果を受け取る変数キー。 |
| `output_value_key` | String | (Optional) この処理結果を格納する変数キー。 |

### Command Type Definition

以下に主要な Command Type とその固有フィールドを定義します。

#### A. Transition / Move Operations
カードの移動に関するコマンド群。`MOVE_CARD`, `DESTROY`, `DISCARD` 等の統合先。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `TRANSITION` | `from_zone` (Opt), `to_zone` (Req), `amount` (Opt) | 汎用移動。ゾーン間移動の基本形。 |
| `DESTROY` | `amount` (Opt), `target_group` (Req) | 破壊（墓地送り）の明示的エイリアス。 |
| `DISCARD` | `amount` (Opt), `target_group` (Req) | 手札破棄の明示的エイリアス。 |
| `MANA_CHARGE` | `amount` (Opt), `target_group` (Req) | マナチャージの明示的エイリアス。 |
| `RETURN_TO_HAND` | `amount` (Opt), `target_group` (Req) | バウンス（手札戻し）の明示的エイリアス。 |
| `DRAW_CARD` | `amount` (Req), `from_zone` (Def: DECK), `to_zone` (Def: HAND) | ドロー操作。 |

**共通フィールド**: `target_group` (Scope), `target_filter` (FilterDef)

#### B. State Mutation / Keywords
状態変更、パワー修正、キーワード付与など。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `MUTATE` | `mutation_kind` (Req), `amount` (Opt), `str_param` (Opt) | 汎用変異。`COST`, `POWER_MOD`, `POWER_SET`, `HEAL` 等を指定。 |
| `ADD_KEYWORD` | `mutation_kind` (Req: Keyword), `amount` (Opt) | キーワード能力の付与（`GRANT_KEYWORD`）。 |
| `TAP` | `target_group`, `target_filter` | タップ状態にする。 |
| `UNTAP` | `target_group`, `target_filter` | アンタップ状態にする。 |
| `SHIELD_BURN` | `amount`, `target_group` | シールド焼却（墓地送り）。 |

#### C. Query / Measurement
情報の取得、カウント、選択。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `QUERY` | `str_param` (Req), `target_group`, `target_filter` | `CARDS_MATCHING_FILTER`, `OPPONENT_DRAW_COUNT` 等の測定。 |
| `CHOICE` | `amount` (Opt), `flags` (["ALLOW_DUPLICATES"]) | 選択肢からの選択（`SELECT_OPTION`）。 |
| `SELECT_NUMBER` | `max` (Opt) | 数値選択。 |

#### D. Deck Operations / Complex Effects
デッキ操作や複合効果。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `SEARCH_DECK` | `amount`, `target_filter` | デッキ探索。 |
| `SHUFFLE_DECK` | `target_group` | デッキシャッフル。 |
| `REVEAL_CARDS` | `amount`, `target_group` | カード公開。 |
| `LOOK_AND_ADD` | `look_count`, `add_count`, `rest_zone`, `target_filter` | `LOOK_AND_ADD` 相当。 |
| `MEKRAID` | `look_count`, `max_cost`, `select_count`, `play_for_free`, `rest_zone` | メクレイド効果。 |

#### E. Gameplay Flow / Battle
ゲーム進行、戦闘、プレイ。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `ATTACK_PLAYER` | `instance_id`, `target_player` | プレイヤーへの攻撃。 |
| `ATTACK_CREATURE` | `instance_id`, `target_instance` | クリーチャーへの攻撃。 |
| `BREAK_SHIELD` | `amount`, `target_group` | シールドブレイク。 |
| `RESOLVE_BATTLE` | `target_group` | 戦闘解決処理。 |
| `CAST_SPELL` | `str_val` (Opt), `flags` (["OPTIONAL"]) | 呪文効果解決/発動。 |
| `PLAY_FROM_ZONE` | `from_zone`, `to_zone`, `max_cost`, `str_param` (Hint) | ゾーンからのカードプレイ（踏み倒し含む）。 |
| `FLOW` | `flow_type`, `value` (phase/branch), `branches` | フェーズ移行や条件分岐。 |

#### F. Buffer / Special
バッファ操作や特殊処理。

| Type | Specific Fields | Description |
| :-- | :-- | :-- |
| `LOOK_TO_BUFFER` | `look_count`, `target_filter` | バッファへのルック。 |
| `SELECT_FROM_BUFFER` | `amount`, `flags` | バッファ内選択。 |
| `PLAY_FROM_BUFFER` | `from_zone` (BUFFER), `to_zone`, `max_cost` | バッファからのプレイ。 |
| `SUMMON_TOKEN` | `token_id`, `amount` | トークン生成。 |
| `REGISTER_DELAYED_EFFECT` | `str_val`, `value1` (Duration) | 遅延効果登録。 |

---

## 2. Action to Command Mapping (Conversion Rules)

`ActionConverter.convert` により、従来の `Action` (dict) は上記の Command (dict) へ変換されます。

### Primary Mappings

| Legacy Action Type | Target Command Type | Notes |
| :-- | :-- | :-- |
| `MOVE_CARD` | `TRANSITION`, `DISCARD`, `DESTROY`, `MANA_CHARGE`, `RETURN_TO_HAND` | 宛先/元ゾーンにより自動判別。 |
| `DESTROY`, `DISCARD` etc. | `DESTROY`, `DISCARD` etc. | 基本的にそのまま対応する型へ。`SEND_TO_MANA` -> `TRANSITION(to=MANA_ZONE)`。 |
| `DRAW_CARD` | `DRAW_CARD` | |
| `TAP`, `UNTAP` | `TAP`, `UNTAP` | `MUTATE(TAP/UNTAP)` ではなく専用型へ（または MUTATE へ統合検討中だが現状は維持）。 |
| `COUNT_CARDS`, `GET_GAME_STAT` | `QUERY` | `str_param` に測定タイプを格納。 |
| `APPLY_MODIFIER`, `COST_REDUCTION` | `MUTATE` | `mutation_kind="COST"` 等。 |
| `GRANT_KEYWORD` | `ADD_KEYWORD` | |
| `SELECT_TARGET` | `QUERY` | `target_group="TARGET_SELECT"` |

---

## 3. Unmapped Actions Categorization

`unmapped_actions.json` の分類基準。

### Category A: Engine Execution (実行系)
ゲームエンジン（C++側 EffectResolver/ActionGenerator）が生成・処理するアクション。これらは **Engine Support (Phase 4)** での Command 化、または **Unified Entry Point (Phase 2)** での `wrap_action` 対応が必要。

- `BLOCK`
- `RESOLVE_BATTLE`
- `RESOLVE_EFFECT`
- `USE_SHIELD_TRIGGER`
- `RESOLVE_PLAY`
- `ATTACK_PLAYER` / `ATTACK_CREATURE`

### Category B: Card Effects / Editor (編集系)
カード効果として記述されるアクション。これらは **ActionConverter (Phase 1)** での変換対応が必要。

- `LOOK_AND_ADD`
- `MEKRAID`
- `REVOLUTION_CHANGE`
- `FRIEND_BURST`
- `REGISTER_DELAYED_EFFECT`
- `SELECT_OPTION`
- `SEARCH_DECK`
- `MOVE_BUFFER_TO_ZONE`
- `PLAY_FROM_ZONE`
- `SEND_SHIELD_TO_GRAVE`

### Category C: Low Priority / Alias
既存の基本コマンドで表現可能、または使用頻度が低いもの。

- `ADD_SHIELD` (TRANSITION to SHIELD_ZONE)
- `SEND_TO_DECK_BOTTOM` (TRANSITION to DECK_BOTTOM)
- `ADD_MANA` (MANA_CHARGE)

---

作成日: 2025-12-27 (Updated)
