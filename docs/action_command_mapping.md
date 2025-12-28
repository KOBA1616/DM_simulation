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

## 4. 統合方針と推奨統合（2025-12-28）

この節は Action→Command 移行において実際に統合を進めるための明確な方針と、統合すべきコマンド群を示します。実装時は下記の設計を第一決定指針とし、互換性確保のためエイリアスを残すことを推奨します。

### 推奨統合一覧
- `TRANSITION` に統合: `DESTROY`, `DISCARD`, `RETURN_TO_HAND`, `SEND_TO_MANA`, `ADD_SHIELD`, `SEND_TO_DECK_BOTTOM`
	- 理由: これらは本質的に「ゾーン間移動」であり、`from_zone`/`to_zone` と `reason`/`alias` で区別可能。内部的には `TRANSITION` をコアにし、短縮エイリアスを提供する。

- `MUTATE` を汎用化: `APPLY_MODIFIER`, `COST_REDUCTION`, `HEAL`(一部), `POWER_MOD`, `POWER_SET`, `SHIELD_BURN`(必要に応じて)
	- 理由: 状態変化は共通のフィールドで安全に表現可能。だが `TAP`/`UNTAP` や `ADD_KEYWORD` は意味論的に専用命令のまま保持することを推奨。

- `QUERY` に統合: `COUNT_CARDS`, `GET_GAME_STAT`, `SELECT_TARGET`（選択の問い合わせ部分）
	- 理由: 情報取得／選択は「問い合わせ→結果」を返す共通パターン。`query_kind` と `flags` で種類を分ける。

- 探索系の整理: `SEARCH_DECK` と `LOOK_AND_ADD` を `SEARCH` 系サブタイプに整理（`LOOK`/`ADD`/`LOOK_AND_ADD`/`MEKRAID`）

- `PLAY_FROM_ZONE` と `PLAY_FROM_BUFFER` は `PLAY` に統合（`from_zone` パラで区別）

- `CHOICE` と `SELECT_NUMBER` の整理: `CHOICE` を汎用化して数値選択を含める（`choice_type`）

### 実装上の注意点（要約）
- `TRANSITION` をコアにした場合、`reason` と `to_zone` によってトリガやログを分岐させること。既存処理（破壊・破棄・マナ等）は内部フラグで扱う。
- `MUTATE` は `mutation_kind` を明瞭に列挙し、`duration`/`stacking` などの付帯情報を持たせる。
- `QUERY` は副作用を持たせない設計とし、結果の受け渡しは `output_value_key` を通して行う。
- `PLAY` は `from_zone` を必須パラとして扱い、`buffer` は単に `from_zone=BUFFER` として扱うことで `PLAY_FROM_BUFFER` を不要にする。
- 既存コードとの互換性のため、短期的にはエイリアスコマンド（例: `DESTROY`）を残し、内部で `TRANSITION` を呼ぶ実装を採る。

### 期待効果
- Command 辞書の種類が減ることで `ActionConverter` の変換ロジックが単純化される。
- 共通処理（選択・フィルタ・ゾーン処理・ログ）が集約され、テストの表現が容易になる。

---

上記の統合方針をドキュメントに反映しました。実際のコード変更（`ActionConverter` の実装差し替え、既存呼び出し箇所のリファクタ等）を行う場合、別途パッチを作成できます。

