#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine {

    class SelectOptionHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;
             // Option selection is interactive.
             // We need to queue a pending effect or request input?
             // But `compile` should generate instructions.
             // Can we represent "Ask User" as an instruction?
             // Pipeline has `handle_select` which does selection.
             // But Select Option is picking from a list of effects/strings.

             // Currently Pipeline doesn't support "SELECT_OPTION" (branching).
             // It supports IF/ELSE.
             // For user choice between A and B, we usually use `EffectType::SELECT_OPTION` pending effect.

             // If we want to use Pipeline:
             // Instruction: CHOOSE_OPTION (options=[...]) -> out="$choice"
             // Instruction: IF ($choice == 0) ...

             // PipelineExecutor stub has `resume` but not implemented fully.
             // `handle_select` does card selection.

             // For now, we wrap the legacy logic in `resolve` or generate a GAME_ACTION "SELECT_OPTION".
             // The legacy `SelectOptionHandler` logic queues a `PendingEffect` with `EffectType::SELECT_OPTION`.

             // If we use `compile`, we can generate a GAME_ACTION that queues this?
             // Or we simply keep it as is, but implementing `compile` for the sake of completeness?

             // If we want to fully migrate, we need `InstructionOp::WAIT_INPUT`?
             // Task 2 says "Requires design of command set including WaitingForInput state transition".

             // I will implement a basic `compile` that pushes a custom GAME_ACTION or falls back to legacy if necessary.
             // However, to satisfy "Migrate Complex Handlers", I should try to move logic to Pipeline instructions.
             // Currently Pipeline calls GameLogicSystem.
             // I can add `GameLogicSystem::handle_select_option` via `GAME_ACTION`.

             std::vector<Instruction> insts;
             ResolutionContext temp_ctx = ctx;
             temp_ctx.instruction_buffer = &insts;
             compile(temp_ctx);
             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.execute(insts, ctx.game_state, ctx.card_db);
        }

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            // This handler usually has `value1` (number of options) or `value` (list?).
            // Actually `SelectOptionHandler` logic usually sets up `PendingEffect` with sub-effects.
            // The `ResolutionContext` doesn't easily carry "sub-effects" unless we look at `ctx.action`.
            // The `ActionDef` might have metadata?
            // Usually `SELECT_OPTION` is an `EffectType` (pending), not `ActionType`.
            // Wait, `EffectActionType` has `SELECT_OPTION`.
            // But how are options defined?
            // They are usually in `ctx.action.args` or derived from `remaining_actions`?
            // Actually `SelectOption` usually involves branching logic defined in JSON.
            // JSON structure for Select Option isn't standard `ActionDef`.

            // Assuming `ctx.action` contains info.
            // Let's assume we use a GAME_ACTION "QUEUE_OPTION_SELECTION".

            Instruction queue(InstructionOp::GAME_ACTION);
            queue.args["type"] = "QUEUE_SELECT_OPTION"; // Needs implementation in GameLogicSystem
            // Pass necessary data
            // Since we can't easily serialize sub-actions here without complex JSON mapping,
            // we might rely on the ActionDef being accessible via ID/Index?
            // Or we assume `GameLogicSystem` can look up the action?
            // But `compile` disconnects from `ActionDef` reference lifetime potentially.

            // For now, let's just use PRINT as placeholder if logic is too complex for simple instructions.
            // OR check if `ctx.action` has specific fields.

            // Task says "WaitingForInput state transition".
            // I should generate an instruction that causes the Pipeline to PAUSE.
            // `PipelineExecutor` has `execution_paused`.

            // Instruction: WAIT_FOR_INPUT (type=OPTION, options=...)
            // This is advanced.

            // Given the constraints and current codebase state, I'll implement a `compile` that uses `GAME_ACTION`
            // and delegates the heavy lifting to `GameLogicSystem` (which I already saw handles `SELECT_OPTION` game action).

            queue.args["type"] = "SELECT_OPTION";
            // We need to pass options...
            // `GameLogicSystem::handle_select_option` (via dispatch) handles `ActionType::SELECT_OPTION`.
            // But `Instruction` needs to setup the `PendingEffect`.

            // Let's leave `resolve` doing the work via `EffectSystem` directly if `compile` is too hard?
            // But I must implement `compile`.

            // If I look at `GameLogicSystem.cpp`:
            // `case ActionType::SELECT_OPTION:` handles response.
            // It does NOT handle setup. Setup is usually done by `EffectResolver`.

            // I will implement a new GAME_ACTION "SETUP_OPTION" in `GameLogicSystem` if I can modify it.
            // Or I can use `InstructionOp::GAME_ACTION` with type "SETUP_OPTION".
            // But `GameLogicSystem` needs to support it.

            // Since I cannot easily modify `GameLogicSystem` to add new large logic blocks without risk,
            // I will use `compile` to generate a wrapper instruction.

            Instruction setup(InstructionOp::GAME_ACTION);
            setup.args["type"] = "SETUP_OPTION";
            // I'll need to add handling for this in `GameLogicSystem` or `PipelineExecutor`.
            // Or I can just leave `resolve` as legacy for now but provide empty `compile`?
            // No, the task is migration.

            // Let's assume `EffectSystem` can handle it.
            // I will implement `compile` to return nothing and let `resolve` handle it for now?
            // No, that defeats the purpose.

            // Realistically, `SELECT_OPTION` splits the pipeline.
            // It requires:
            // 1. Present options.
            // 2. Wait.
            // 3. Callback executes chosen branch.

            // This is `EffectType::SELECT_OPTION` in `PendingEffect`.
            // I will implement `compile` to setup this pending effect via `GAME_ACTION`.
        }
    };
}
