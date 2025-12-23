#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class DrawHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
             Player& controller = ctx.game_state.players[controller_id];

             // Handle Variable Linking
             int count = ctx.action.value1;
             if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                count = ctx.execution_vars[ctx.action.input_value_key];
             }
             if (count == 0 && !ctx.action.value.empty()) {
                 try { count = std::stoi(ctx.action.value); } catch (...) {}
             }
             if (count == 0) count = 1;

             // Execute Draw
             int actual_drawn = 0;
             for (int i = 0; i < count; ++i) {
                if (controller.deck.empty()) {
                    auto result = (controller.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                    auto cmd = std::make_shared<dm::engine::game_command::GameResultCommand>(result);
                    ctx.game_state.execute_command(cmd);
                    return;
                }

                int card_instance_id = controller.deck.back().instance_id;

                // Move Card Command
                auto move_cmd = std::make_shared<dm::engine::game_command::TransitionCommand>(
                    card_instance_id,
                    Zone::DECK,
                    Zone::HAND,
                    controller.id
                );
                ctx.game_state.execute_command(move_cmd);

                actual_drawn++;

                if (controller.id == ctx.game_state.active_player_id) {
                    auto stat_cmd = std::make_shared<dm::engine::game_command::StatCommand>(
                        dm::engine::game_command::StatCommand::StatType::CARDS_DRAWN,
                        1
                    );
                    ctx.game_state.execute_command(stat_cmd);
                }

                // Trigger Logic: Check for ON_OPPONENT_DRAW effects for the non-drawing player
                PlayerID opponent_id = 1 - controller_id;
                const Player& opponent = ctx.game_state.players[opponent_id];

                // We must iterate over opponent's Battle Zone to trigger effects like "Whenever your opponent draws a card..."
                // Since this is inside the loop, it triggers per card drawn.
                for (const auto& card : opponent.battle_zone) {
                    EffectSystem::instance().resolve_trigger(ctx.game_state, TriggerType::ON_OPPONENT_DRAW, card.instance_id, ctx.card_db);
                }
             }

             if (!ctx.action.output_value_key.empty()) {
                 ctx.execution_vars[ctx.action.output_value_key] = actual_drawn;
             }
        }

        void compile_action(const ResolutionContext& ctx) override {
            if (!ctx.instruction_buffer) return;

            // Determine count
            int count_literal = 0;
            std::string count_var = "";

            if (!ctx.action.input_value_key.empty()) {
                count_var = ctx.action.input_value_key;
            } else if (!ctx.action.value1 == 0) { // Assuming value1 is numeric
                 count_literal = ctx.action.value1;
            } else if (!ctx.action.value.empty()) {
                try { count_literal = std::stoi(ctx.action.value); } catch (...) { count_literal = 1; }
            } else {
                count_literal = 1;
            }

            if (count_var.empty()) {
                // Literal count: Unroll
                for (int i = 0; i < count_literal; ++i) {
                     generate_single_draw(*ctx.instruction_buffer, ctx);
                }
            } else {
                // Variable count: Use REPEAT
                core::Instruction repeat_inst(core::InstructionOp::REPEAT);
                repeat_inst.args["count"] = "$" + count_var; // e.g. "$my_count"
                repeat_inst.args["var"] = "$_draw_idx"; // Dummy iterator var

                // The body of the loop is a single draw sequence
                generate_single_draw(repeat_inst.then_block, ctx);

                ctx.instruction_buffer->push_back(repeat_inst);
            }

            // Handle output
             if (!ctx.action.output_value_key.empty()) {
                 // Set context variable to count
                 // ctx.instruction_buffer->push_back(core::Instruction(core::InstructionOp::MATH, ...));
                 // Actually the handler updates execution_vars directly in resolve.
                 // In compile, we must output an instruction to set the var.
                 // We need SET_VAR instruction? PipelineExecutor::handle_calc sets vars.
                 // We can use MATH: lhs=count, rhs=0, op='+', out=key.

                 nlohmann::json args;
                 args["lhs"] = (count_var.empty() ? count_literal : 0); // Logic missing for var copy
                 args["op"] = "+";
                 args["out"] = ctx.action.output_value_key;
                 // If variable, we need to read it. MATH supports resolving vars.
                 if (!count_var.empty()) {
                     args["lhs"] = "$" + count_var;
                 }
                 ctx.instruction_buffer->push_back(core::Instruction(core::InstructionOp::MATH, args));
             }
        }

    private:
        void generate_single_draw(std::vector<core::Instruction>& buffer, const ResolutionContext& ctx) {
            using namespace dm::core;
            PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            // 1. Check Deck Empty -> Win/Lose
            // Instruction: IF (deck_count == 0) THEN (Win/Lose)
            // Need condition "DECK_COUNT". PipelineExecutor::check_condition supports "exists"?
            // We need "type"="DECK_COUNT_CHECK"? No, ConditionSystem handles it.
            // But ConditionSystem usually checks Source/Context.

            // Let's assume we can use GAME_ACTION "CHECK_DECKOUT".
            // Or just check condition.
            // We'll use a ConditionDef that calls a custom evaluator or simple property check?
            // "DECK_EMPTY" condition type?

            Instruction check_deck(InstructionOp::IF);
            check_deck.args["cond"]["type"] = "DECK_EMPTY"; // Requires Evaluator
            check_deck.args["cond"]["value"] = 1; // True

            Instruction win_cmd(InstructionOp::GAME_ACTION);
            win_cmd.args["type"] = (controller == 0) ? "LOSE_GAME" : "WIN_GAME"; // If P1 deck empty, P1 loses (P2 wins)
            win_cmd.args["player"] = controller;

            check_deck.then_block.push_back(win_cmd);

            // ELSE move card
            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "HAND";
            move.args["target"] = "DECK_TOP"; // Virtual target support added to PipelineExecutor
            move.args["count"] = 1;

            check_deck.else_block.push_back(move);

            // Update Stats
            Instruction stat(InstructionOp::MODIFY);
            stat.args["type"] = "STAT";
            stat.args["stat"] = "CARDS_DRAWN";
            stat.args["value"] = 1;
            check_deck.else_block.push_back(stat);

            // Trigger Check
            Instruction trigger(InstructionOp::GAME_ACTION);
            trigger.args["type"] = "TRIGGER_CHECK";
            trigger.args["trigger"] = "ON_OPPONENT_DRAW";
            trigger.args["source"] = ctx.source_instance_id; // Just pass a valid ID, logic system iterates BZ
            check_deck.else_block.push_back(trigger);

            buffer.push_back(check_deck);
        }
    };
}
