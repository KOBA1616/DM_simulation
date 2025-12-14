#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/target_utils.hpp"

namespace dm::engine {

    class DrawHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
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
                    ctx.game_state.winner = (controller.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                    return;
                }
                CardInstance c = controller.deck.back();
                controller.deck.pop_back();
                controller.hand.push_back(c);
                actual_drawn++;

                if (controller.id == ctx.game_state.active_player_id) {
                    ctx.game_state.turn_stats.cards_drawn_this_turn++;
                }

                // Trigger Logic: Check for ON_OPPONENT_DRAW effects for the non-drawing player
                PlayerID opponent_id = 1 - controller_id;
                const Player& opponent = ctx.game_state.players[opponent_id];

                // We must iterate over opponent's Battle Zone to trigger effects like "Whenever your opponent draws a card..."
                // Since this is inside the loop, it triggers per card drawn.
                for (const auto& card : opponent.battle_zone) {
                    GenericCardSystem::resolve_trigger(ctx.game_state, TriggerType::ON_OPPONENT_DRAW, card.instance_id, ctx.card_db);
                }
             }

             if (!ctx.action.output_value_key.empty()) {
                 ctx.execution_vars[ctx.action.output_value_key] = actual_drawn;
             }
        }
    };
}
