# Action -> Command Complete Migration Guide (C++ source of truth)

Purpose
- Provide a step-by-step, low-risk plan to fully migrate from Action-based flows to Command-first flows.
- C++ is the single source of truth. Python is minimized and ideally removed from core execution.
- New files are introduced first. Existing files are changed only during final cutover.

Scope
- Engine action generation, AI, training, tools, and tests.
- Python compatibility layers and wrappers.

Non-goals
- No behavioral changes to game rules.
- No immediate removal of legacy Action code without replacement.

Current implementation (confirmed)
- C++ legal action generation is in IntentGenerator and is currently the primary logic source.
  - [src/engine/actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp#L1-L120)
- Python ActionGenerator is a fallback with minimal logic.
  - [dm_ai_module.py](dm_ai_module.py#L590-L640)
- Python bindings alias ActionGenerator -> IntentGenerator (C++).
  - [src/bindings/bind_engine.cpp](src/bindings/bind_engine.cpp#L260-L286)
- Engine uses actions internally for decision and progression.
  - [src/engine/game_instance.cpp](src/engine/game_instance.cpp#L52-L120)
  - [src/engine/systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp#L270-L307)
- Command system is implemented in C++ (execution side).
  - [src/engine/systems/command_system.cpp](src/engine/systems/command_system.cpp#L1-L120)
- Command types and high-level command classes exist in C++.
  - [src/engine/game_command/game_command.hpp](src/engine/game_command/game_command.hpp#L1-L60)
  - [src/engine/game_command/action_commands.hpp](src/engine/game_command/action_commands.hpp#L1-L90)
- Python command-first pipeline exists but still pulls Actions as source.
  - [dm_toolkit/commands.py](dm_toolkit/commands.py#L1-L78)
  - [dm_toolkit/unified_execution.py](dm_toolkit/unified_execution.py#L1-L120)
  - [dm_toolkit/action_to_command.py](dm_toolkit/action_to_command.py#L1-L120)

Constraints and policy
- Add new files first. Do not modify existing files until the final cutover phase.
- C++ is authoritative for generation and execution.
- Python must be thin; ideally no Action generation in Python.
- Plan must be usable by low-resource models; keep steps small with explicit validation.

Target architecture (end state)
- C++ generates CommandDef objects directly as the authoritative output.
- Python receives CommandDef objects only; Action objects are no longer produced for normal flows.
- Logging/serialization uses CommandDef.to_dict() on demand (no dicts as primary output).
- dm_toolkit.commands calls a native command generator (not ActionGenerator).
- ActionGenerator remains only as a compatibility shim or is removed at the end.

CommandDef.to_dict() specification (logging/serialization only)
- Purpose: Provide a stable, minimal command representation for logs, tests, and telemetry.
- Not used as the primary execution input.

Required keys
- type: string
- uid: string (unique per command instance; stable for logging)

Optional keys (present only when relevant)
- instance_id: int
- source_instance_id: int
- target_instance_id: int
- owner_id: int
- player_id: int
- from_zone: string
- to_zone: string
- target_group: string
- target_filter: object
- amount: int
- value1: int
- value2: int
- str_param: string
- flags: array
- slot_index: int
- target_slot_index: int
- optional: bool
- up_to: bool
- query_type: string
- options: array

Type normalization rules
- type: Use canonical CommandType names (e.g., "PLAY_CARD", "ATTACK", "PASS_TURN").
- from_zone/to_zone: Use canonical zone names (e.g., "HAND", "MANA", "BATTLE").
- target_filter: JSON-friendly structure with zones/types/owner/count where applicable.
- Only include fields that are meaningful for the specific command type.

Example output (PLAY_CARD)
{
  "type": "PLAY_CARD",
  "uid": "c2f8e4a8-3b64-4d4a-9c2c-9f820e9c2a10",
  "instance_id": 12345,
  "owner_id": 0
}

Example output (ATTACK)
{
  "type": "ATTACK",
  "uid": "8d777b11-7b07-402e-b6ce-6a3e64de2b74",
  "source_instance_id": 222,
  "target_instance_id": 777,
  "target_player": 1
}

CommandDef.to_dict() C++ implementation guide
- Implement to_dict() on CommandDef (or a helper serializer) in C++ only.
- Ensure to_dict() is available via pybind so Python can log without custom adapters.
- Do not use to_dict() as an execution path.

Recommended approach (minimal, consistent)
- Add a C++ method CommandDef::to_dict() that returns a py::dict in bindings.
- Use a single shared helper that normalizes keys and omits irrelevant fields.

Placement
- If CommandDef is a struct/class in C++: add a member function to_dict() in its header/impl.
- If CommandDef is a binding-only wrapper: implement a free function in bindings and expose it.

Field mapping rules
- Always emit: type, uid.
- Emit instance_id/source_instance_id/target_instance_id only when > 0.
- Emit from_zone/to_zone only when non-empty.
- Emit target_filter only when populated.
- Emit flags/optional/up_to only when used.
- Emit query_type/options only for QUERY/DECIDE flows.

Serialization example (pseudo C++)
py::dict to_dict(const CommandDef& cmd) {
    py::dict d;
    d["type"] = command_type_to_string(cmd.type);
    d["uid"] = cmd.uid;
    if (cmd.instance_id > 0) d["instance_id"] = cmd.instance_id;
    if (cmd.source_instance_id > 0) d["source_instance_id"] = cmd.source_instance_id;
    if (cmd.target_instance_id > 0) d["target_instance_id"] = cmd.target_instance_id;
    if (!cmd.from_zone.empty()) d["from_zone"] = cmd.from_zone;
    if (!cmd.to_zone.empty()) d["to_zone"] = cmd.to_zone;
    if (cmd.amount != 0) d["amount"] = cmd.amount;
    if (!cmd.str_param.empty()) d["str_param"] = cmd.str_param;
    if (cmd.optional) d["optional"] = true;
    if (cmd.up_to) d["up_to"] = true;
    return d;
}

Binding note
- Expose to_dict() via pybind on CommandDef, not on the generator.
- Keep conversion centralized to avoid diverging dict shapes.

Phase 0: Inventory and stability baseline (no code changes)
1) Inventory all Action generation call sites
   - C++: IntentGenerator usage in engine/ai/flow
     - [src/engine/actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp#L1-L120)
     - [src/engine/game_instance.cpp](src/engine/game_instance.cpp#L52-L120)
     - [src/engine/systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp#L270-L307)
     - [src/ai/self_play/self_play.cpp](src/ai/self_play/self_play.cpp#L60-L90)
   - Python fallback ActionGenerator
     - [dm_ai_module.py](dm_ai_module.py#L590-L640)
   - Python tools/tests training using Action or IntentGenerator
     - [test_without_gameinstance.py](test_without_gameinstance.py#L18-L60)
     - [test_turn_ending.py](test_turn_ending.py#L28-L70)
     - [training/head2head.py](training/head2head.py#L402-L442)
     - [training/fine_tune_with_mask.py](training/fine_tune_with_mask.py#L28-L80)
     - [tools/emit_play_attack_states.py](tools/emit_play_attack_states.py#L17-L50)

2) Baseline behavior capture
   - Record: action counts per phase, and key decision points (PASS, MANA, PLAY, ATTACK).
   - Save logs as reference before migration.

Validation checklist
- Can you enumerate at least one Action call site in each category (engine, AI, training, tools, tests)?
- Can you reproduce current legal action counts in a sample game?

Phase 1: Create new C++ command generator (new files only)
Goal: Introduce a Command-first generator without touching existing IntentGenerator.

New C++ files
- src/engine/commands/command_generator.hpp (new)
- src/engine/commands/command_generator.cpp (new)

Status: IMPLEMENTED (Phase 1 v1 bridge)
- Created: src/engine/commands/command_generator.hpp
- Created: src/engine/commands/command_generator.cpp

Design
- Input: GameState, CardDB.
- Output: vector<CommandDef> only.
- Logic: reuse IntentGenerator internal strategies or call existing action flow and map to CommandDef in C++.

Recommended approach
- Step 1: CommandGenerator calls IntentGenerator to get Actions, then maps to CommandDef in C++ (temporary bridge).
- Step 2: Replace mapping with direct command generation strategy by strategy.

Validation checklist
- New generator compiles but is not yet used by production code.
- Simple test function returns non-empty commands in MAIN phase.

Phase 2: Expose command generator via bindings (new files + minimal glue later)
Goal: Make Command-first generation accessible to Python without modifying existing call paths.

New binding helper
- src/bindings/bind_command_generator.cpp (new)

Status: IMPLEMENTED (Phase 2)
- Created: src/bindings/bind_command_generator.hpp
- Created: src/bindings/bind_command_generator.cpp

Binding API
- dm_ai_module.generate_commands(state, card_db) -> list of CommandDef objects
- Optional: CommandDef.to_dict() for logging/serialization only

Validation checklist
- Python can import dm_ai_module.generate_commands without touching ActionGenerator.
- Returned CommandDef objects can be encoded by CommandEncoder without Action-to-Command conversion.

Phase 3: Python command-first API (new module only)
Goal: Add a new module that is command-first, without touching old modules.

New Python module
- dm_toolkit/commands_v2.py (new)

Status: IMPLEMENTED (Phase 3)
- Created: dm_toolkit/commands_v2.py

Behavior
- Prefer dm_ai_module.generate_commands (CommandDef-only).
- If unavailable, optional fallback to old generate_legal_actions (temporary).
- Provide strict mode: error if command generator missing.

Validation checklist
- commands_v2.generate_legal_commands returns the same (or compatible) command count for test states.

Phase 4: Add a parallel execution pipeline (new module only)
Goal: Execute commands without Action wrappers.

New Python module
- dm_toolkit/unified_execution_v2.py (new)

Status: IMPLEMENTED (Phase 3)
- Created: dm_toolkit/unified_execution_v2.py

Behavior
- Accept CommandDef objects only.
- Use EngineCompat.ExecuteCommand with strict validation.
- Do not accept Action-like objects.
- Only convert to dict for logs via CommandDef.to_dict().

実装状況（最新）
-----------------
- 日付: 2026-02-09
- 概要: Python 側の残存レガシー呼び出しのバッチ移行を継続し、`training/head2head.py` の複数箇所でコマンドファースト生成を優先するよう更新しました。これによりトレーニングループがまずネイティブの `commands_v2.generate_legal_commands(..., strict=False)` を試し、失敗時に従来の `IntentGenerator.generate_legal_commands` にフォールバックします。
- 変更ファイル:
  - [training/head2head.py](training/head2head.py#L380-L430)
  - [training/head2head.py](training/head2head.py#L960-L1010)
  - [training/head2head.py](training/head2head.py#L1060-L1090)
- 検証: 直近の parity 単体テストを実行し、全テストがパスしました（`1 passed` を確認）。
- 次ステップ: 残存する呼び出し箇所を 10–20 ファイルずつのバッチで順次移行し、各バッチ後に parity テストと主要テスト群を実行します。

追記 — バッチ(次の 10 ファイル相当)適用
-------------------------------------
- 日付: 2026-02-09
- 概要: 追加バッチを適用し、コマンド優先化を以下のファイル群に導入しました（既存の ActionGenerator フォールバックは維持）。
- 変更ファイル:
  - [dm_toolkit/training/evolution_ecosystem.py](dm_toolkit/training/evolution_ecosystem.py#L1-L80)
  - [dm_toolkit/ai/analytics/deck_consistency.py](dm_toolkit/ai/analytics/deck_consistency.py#L1-L80)
  - [dm_toolkit/ai/ga/evolve.py](dm_toolkit/ai/ga/evolve.py#L1-L80)
  - [dm_toolkit/gui/ai/mcts_python.py](dm_toolkit/gui/ai/mcts_python.py#L1-L120)
- 検証: parity テストを実行し、1 件のパリティテストが合格しました（`1 passed`）。
- 次ステップ: 残りのレガシー呼び出し箇所をさらにバッチで移行し、各バッチ後に `tests/test_command_migration_parity.py` と主要テスト群を実行します。

残タスクのサマリ
-----------------
- コアバインディング (`dm_ai_module.py`) は現在も一部 `ActionGenerator` を露出しています。これらは互換性のために残しますが、最終フェーズでは `dm_ai_module.generate_commands` を優先的に提供し、`ActionGenerator` を互換 shim に限定する予定です。
- `dm_toolkit/engine/compat.py` は移行中の互換レイヤであり、ここでの API を段階的に切り替え・無効化することで Python 全体の切替作業を制御できます。
- `dm_toolkit/gui/headless.py` にあるスタブはテスト環境向けのフォールバックです。ネイティブバインディングが揃うまで保持します。

優先度と次のステップ
-------------------
1. パリティ強化テストを作成（Action->Command のフィールドレベルの一致検証）。
2. コアバインディングの監査: `dm_ai_module.py` 内の Action 露出点を列挙し、`generate_commands` を優先するパッチ案を用意。
3. 小さな試験的切替を 1 箇所のエンジンループで実施（リスク低減のためブランチで実験）。
4. 全面切替のためのロールアウト計画（バージョンやフラグで段階的適用）。

注: これらの作業は必ず小さな変更単位で実施し、各ステップごとに parity テストと主要テスト群を実行してください。

Validation checklist
- A CommandDef produced by commands_v2 executes successfully.

Phase 5: Dual-run verification (new test harness)
Goal: Compare Action-based and Command-based outputs for equality.

New tests
- tests/test_command_migration_parity.py (new)

Checks
- For a fixed seed and state snapshots:
  - ActionGenerator -> map_action -> command list
  - CommandGenerator -> command list
  - Compare counts, types, and index encodings.

Validation checklist
- Parity test passes for at least MANA, MAIN, ATTACK phases.

Phase 6: Update call sites to command-first (final cutover)
Goal: Replace all references to Action-based flows.

Expected edits (to be done only after previous phases are green)
- Python tools and training
  - Replace EngineCompat.ActionGenerator_generate_legal_commands with commands_v2
  - Files:
    - [training/head2head.py](training/head2head.py#L402-L442)
    - [training/fine_tune_with_mask.py](training/fine_tune_with_mask.py#L28-L80)
    - [tools/emit_play_attack_states.py](tools/emit_play_attack_states.py#L17-L50)
- Tests
  - Replace ActionGenerator usage with commands_v2
    - [test_without_gameinstance.py](test_without_gameinstance.py#L18-L60)
    - [test_turn_ending.py](test_turn_ending.py#L28-L70)
- Core Python API
  - Switch dm_toolkit.commands to delegate to commands_v2
    - [dm_toolkit/commands.py](dm_toolkit/commands.py#L1-L78)
- C++ engine flow (optional but recommended)
  - Replace IntentGenerator in engine loops with CommandGenerator
    - [src/engine/game_instance.cpp](src/engine/game_instance.cpp#L52-L120)
    - [src/engine/systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp#L270-L307)

Validation checklist
- All training scripts and tools run without Action imports.
- All tests pass without Action-based generation.

Phase 7: Decommission Action (cleanup)
Goal: Remove Action usage from main code paths.

Actions
- Keep ActionGenerator only for legacy fallback or delete entirely after stable release.
- Reduce Python fallback in dm_ai_module.py to minimal stubs or remove if native always available.
  - [dm_ai_module.py](dm_ai_module.py#L590-L640)

Validation checklist
- No grep hits for ActionGenerator in production paths.
- No runtime errors when dm_ai_module is native-only.

Mapping notes (Action -> Command)
- Action is a player intent token; Command is engine-executable.
- Command generation should be the authoritative source to avoid dual logic.
- Use C++ CommandType in game_command.hpp for canonical types.
  - [src/engine/game_command/game_command.hpp](src/engine/game_command/game_command.hpp#L1-L60)

Recommended minimal C++ command generator behavior (v1)
- PASS -> PASS_TURN
- MANA_CHARGE -> MANA_CHARGE
- DECLARE_PLAY or PLAY_CARD -> PLAY_CARD
- ATTACK_* -> ATTACK (with target fields)
- SELECT_TARGET/SELECT_OPTION -> DECIDE or QUERY response (align with CommandSystem)

Risks and mitigations
- Risk: Mismatch between Action and Command semantics.
  - Mitigation: parity tests in Phase 5.
- Risk: Python tooling still expects Actions.
  - Mitigation: provide commands_v2 and gradually update call sites.
- Risk: UI expects Action enums.
  - Mitigation: keep UI mapping layer or convert GUI to display commands.

Low-resource step-by-step checklist
- Step A: Add new C++ CommandGenerator files. Build. No usage change.
- Step B: Add new binding file to expose generate_commands. Build. Validate in Python.
- Step C: Add commands_v2.py and unified_execution_v2.py. Write small usage tests.
- Step D: Add parity test. Basic parity existence test added as tests/test_command_migration_parity.py
- Step D: Add parity test. Fix differences until parity is stable.
- Step E: Switch Python tools/training/tests to commands_v2.
- Step F: Switch engine loops to CommandGenerator (optional but ideal).
- Step G: Remove ActionGenerator usage from production paths.

Open questions to resolve early
- Should C++ implement direct command generation per phase or map from Actions initially?
- What is the cutoff for removing Python fallback in dm_ai_module.py?

Appendix: Evidence of Action usage
- IntentGenerator generates actions in C++.
  - [src/engine/actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp#L1-L120)
- Python fallback ActionGenerator exists.
  - [dm_ai_module.py](dm_ai_module.py#L590-L640)
- Python command wrapper still pulls ActionGenerator.
  - [dm_toolkit/commands.py](dm_toolkit/commands.py#L1-L78)
- Engine uses actions in loops.
  - [src/engine/game_instance.cpp](src/engine/game_instance.cpp#L52-L120)
  - [src/engine/systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp#L270-L307)
- Training and tools call command generation wrappers.
  - [training/head2head.py](training/head2head.py#L402-L442)
  - [training/fine_tune_with_mask.py](training/fine_tune_with_mask.py#L28-L80)
  - [tools/emit_play_attack_states.py](tools/emit_play_attack_states.py#L17-L50)

## 実装状況（最新）

- 日付: 2026-02-08
- ステータス概要: Phase 1〜3 完了、Phase 4 実装済み、Phase 5 基本パリティ検査を追加して合格、Phase 6 の Python 側呼び出し置換を段階的に実施中。
- ビルド: Release ビルドでネイティブ拡張生成に成功（出力例: `bin/Release/dm_ai_module.cp312-win_amd64.pyd`, `bin/Release/dm_core.lib`）。

主要変更ファイル（抜粋）
- C++ (新規/変更)
  - [src/engine/commands/command_generator.hpp](src/engine/commands/command_generator.hpp#L1-L120)
  - [src/engine/commands/command_generator.cpp](src/engine/commands/command_generator.cpp#L1-L200)
  - [src/bindings/bind_command_generator.hpp](src/bindings/bind_command_generator.hpp#L1-L80)
  - [src/bindings/bind_command_generator.cpp](src/bindings/bind_command_generator.cpp#L1-L180)
  - `CMakeLists.txt` にバイディングと実装ソースを追加
- Python (新規/変更)
  - [dm_toolkit/commands_v2.py](dm_toolkit/commands_v2.py#L1-L160) （コマンド優先の薄いラッパー）
  - [dm_toolkit/unified_execution_v2.py](dm_toolkit/unified_execution_v2.py#L1-L200)
  - [dm_toolkit/gui/input_handler.py](dm_toolkit/gui/input_handler.py#L1-L220)（破損版を修復して `commands_v2` を利用）
  - [tests/test_command_migration_parity.py](tests/test_command_migration_parity.py#L1-L120)（基本パリティ検査、現在 1 件のテストが通過）

進行中の作業
- 残りの Python 呼び出し箇所を段階的に `commands_v2` に置換中（レガシー呼び出しの検索で多数ヒット、バッチで置換し都度テストを実行する方針）。
- Phase 5 を拡張して「Action ↔ Command の内容同値検査」を充実させる予定（現在は基本的な存在/カウント比較テスト）。

次の推奨アクション
- 1) 残りの呼び出し箇所を 10〜20 ファイル単位でバッチ置換し、各バッチごとにパリティテストと既存テストを実行。
- 2) パリティテストを拡張してフィールド単位の同値性チェックを追加。
- 3) 十分なパリティが確認できたらエンジンループを `IntentGenerator` から `CommandGenerator` へ切替（Phase 6 の最終化）。

注記: 重要な設計判断や低レベルの整合性は C++ 側が最終的な根拠です。Python 側の変更は段階的かつ後方互換重視で行っています。
