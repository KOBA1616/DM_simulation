# Action → Command マッピング

このファイルは `dm_ai_module.py` の `ActionType` / `EffectActionType` と、現状存在する `Command` 実装との対応をまとめたものです。

## 要約
- コア実行は既に `Command` クラス群で実装されており、`Action` は `action_to_command()` / `ActionCommandWrapper` を通して `Command` として実行可能。
- 移行の主作業は未マップの `ActionType` / `EffectActionType` を網羅的に `Command` に対応させ、AI/GUI/engine 層を段階的にコマンドベースへ切替えること。

## ActionType -> Command マッピング

- `PLAY_CARD`: 構文上のプレイ系。一部シナリオでは `CastSpellCommand` または `PlayFromZoneCommand` に相当（文脈依存）。
- `PLAY_CARD_INTERNAL`: 内部専用プレイフロー（明示的な Command 実装なし、システム内でハンドリング）。
- `ATTACK_CREATURE`: `AttackCreatureCommand`
- `ATTACK_PLAYER`: `AttackPlayerCommand`
- `BLOCK`: （既定のブロック解決は `EffectResolver` 内で実施、専用 Command クラスなし）
- `USE_SHIELD_TRIGGER`: （EffectResolver / pending effects 経由で処理）
- `RESOLVE_EFFECT`: pending effect を取り出して解決（`EffectResolver.resolve_action` による処理）
- `RESOLVE_PLAY`: `ResolvePlayCommand`（`ActionGenerator` は `a.command = action_to_command(ResolvePlayCommand(...))` を設定）
- `RESOLVE_BATTLE`: 戦闘解決は `EffectResolver` / `GenericCardSystem` 側で処理
- `BREAK_SHIELD`: `BreakShieldCommand`（存在）
- `SELECT_TARGET`: `SelectTargetCommand`
- `USE_ABILITY`: `UseAbilityCommand`
- `DECLARE_REACTION`: `DeclareReactionCommand`
- `DECLARE_PLAY`: `DeclarePlayCommand`
- `PAY_COST`: `PayCostCommand`
- `MANA_CHARGE`: `ManaChargeCommand`
- `PASS`: `PassCommand`

(注) `ActionGenerator.generate_legal_actions()` は上記多くの `a.command` を作成している。

## EffectActionType -> 実行手段（Command / Pipeline）

EffectActionType の多くは `GenericCardSystem.resolve_effect` / `_generic_resolve_effect_with_targets` / `PipelineExecutor` によって実行される。コマンド実装が直接存在するものとカテゴリ分けすると下記の通り。

- 明示的な Command クラスで扱われる要素:
  - トランジション系 (`TRANSITION`) -> `TransitionCommand`
  - マナ/移動/変化系 (`ADD_MANA`, `SEND_TO_MANA`, `SEND_TO_DECK_BOTTOM`, `SEARCH_DECK`, `SEARCH_DECK_BOTTOM`, `PLAY_FROM_ZONE` 等) -> 主に `PipelineExecutor` の `Instruction` または `TransitionCommand` / `MutateCommand` による処理
  - タップ/アンタップ (`TAP`, `UNTAP`) -> `MutateCommand` / `MutationType` を通じた処理
  - `RETURN_TO_HAND` -> `TransitionCommand` / `_generic_resolve_effect_with_targets` 内の移動処理
  - `BREAK_SHIELD` 相当 -> `BreakShieldCommand` または EffectResolver による処理

- EffectResolver / Pipeline に依存するもの（既存の命令列実行 / 効果解決により処理）:
  - `DRAW_CARD`, `COUNT_CARDS`, `GET_GAME_STAT`, `DESTROY`, `DISCARD`, `REVEAL_CARDS`, `SHUFFLE_DECK`, `ADD_SHIELD`, `CAST_SPELL`, `PUT_CREATURE`, `RESOLVE_BATTLE` 等。これらは `GenericCardSystem.resolve_effect` または `PipelineExecutor` の `Instruction` 群により実装されている。

## 未カバー / 要注意項目

- `PLAY_CARD_INTERNAL`, `BLOCK` など一部アクションは専用の Command クラスを持たず、`EffectResolver` や既存のフローで処理されている。移行時にこれらを Command 化するか、互換層を維持するか方針決定が必要。
- GUI のエディタ (`dm_toolkit/gui/editor/logic_tree.py`) と `dm_ai_action_command_shim.py` によるマッピングロジックの差分を整合する作業が必要。
- `ActionType` と `EffectActionType` のすべてのメンバーに対し、`action_to_command` / `translate_action_to_command` が明示的なマッピングを持つことを保証する必要がある。

## 推奨次ステップ（優先順）
1. `action_to_command` / `dm_ai_action_command_shim.translate_action_to_command` を拡充し、上記未カバーや曖昧箇所を明確にマップする（自動テスト付き）。
2. `ActionGenerator` に新 API `generate_legal_commands()` を追加して、Action ではなく直接 `Command` を生成できるようにする（段階移行を容易にする）。
3. AI 層（MCTS/ActionEncoder）を段階的にコマンドベースへ対応。`action_to_index` を `command_to_index` に拡張・後方互換化。
4. 既存テストをコマンド経由でも実行する統合テストを追加。

---

作成日: 2025-12-27
