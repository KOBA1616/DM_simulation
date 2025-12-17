#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    class DiscardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Handle Random Discard or All Discard if selection is implicit

            // Check scope/target_choice
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectSystem::instance().delegate_selection(ctx);
                 return;
            }

            // Handle RANDOM discard
            // Usually specified by filter.selection_mode or just "random" keyword?
            // ActionDef has `target_choice` which might be "RANDOM".
            // Or FilterDef has `selection_mode`.

            // Target Player
            PlayerID target_pid = ctx.game_state.active_player_id;
            if (ctx.action.target_player == "OPPONENT") {
                target_pid = 1 - ctx.game_state.active_player_id;
            } else if (ctx.action.target_player == "SELF") {
                 target_pid = ctx.game_state.active_player_id;
            } else if (!ctx.action.target_player.empty()) {
                // Could be explicit ID if parsing allowed, but strings are usually relative
            }
            // "scope" also defines target.
            if (ctx.action.scope == TargetScope::PLAYER_OPPONENT) {
                target_pid = 1 - ctx.game_state.active_player_id;
            } else if (ctx.action.scope == TargetScope::PLAYER_SELF) {
                target_pid = ctx.game_state.active_player_id;
            }

            Player& p = ctx.game_state.players[target_pid];
            std::vector<int> discard_candidates;

            // Filter validation
            for (const auto& card : p.hand) {
                if (!ctx.card_db.count(card.card_id)) continue;
                 const auto& def = ctx.card_db.at(card.card_id);
                 // We need a helper to validate filter against a card in a zone
                 // TargetUtils::is_valid_target needs context?
                 if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, ctx.source_instance_id, target_pid)) {
                     discard_candidates.push_back(card.instance_id);
                 }
            }

            int count = ctx.action.filter.count.value_or(1);
            if (ctx.action.value1 > 0) count = ctx.action.value1; // Legacy support

            // If ALL, discard all candidates
            if (ctx.action.filter.selection_mode == "ALL" || ctx.action.target_choice == "ALL") {
                count = discard_candidates.size();
            }

            if (discard_candidates.empty()) {
                // No cards to discard
                if (!ctx.action.output_value_key.empty()) {
                    ctx.execution_vars[ctx.action.output_value_key] = 0;
                }
                return;
            }

            std::vector<int> to_discard;

            if (ctx.action.scope == TargetScope::RANDOM || ctx.action.target_choice == "RANDOM" || ctx.action.filter.selection_mode == "RANDOM") {
                std::shuffle(discard_candidates.begin(), discard_candidates.end(), ctx.game_state.rng);
                int num = std::min((int)discard_candidates.size(), count);
                for (int i = 0; i < num; ++i) to_discard.push_back(discard_candidates[i]);
            } else {
                 // Non-random, non-select (e.g. "ALL")
                 // If not ALL and not RANDOM and not SELECT, what is it?
                 // Maybe "First X"? Or undefined?
                 // Default to ALL if count matches size?
                 int num = std::min((int)discard_candidates.size(), count);
                 for (int i = 0; i < num; ++i) to_discard.push_back(discard_candidates[i]);
            }

            int discarded_count = 0;
            for (int iid : to_discard) {
                // Move from Hand to Graveyard
                auto it = std::find_if(p.hand.begin(), p.hand.end(), [iid](const CardInstance& c){ return c.instance_id == iid; });
                if (it != p.hand.end()) {
                    CardInstance c = *it;
                    p.hand.erase(it);
                    c.is_tapped = false;
                    p.graveyard.push_back(c);
                    discarded_count++;
                }
            }

            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = discarded_count;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;

            int discarded_count = 0;
            for (int tid : *ctx.targets) {
                 // Targets could be in any player's hand if filter allowed it.
                 // We search both players.
                 for (auto& p : ctx.game_state.players) {
                     auto it = std::find_if(p.hand.begin(), p.hand.end(), [tid](const CardInstance& c){ return c.instance_id == tid; });
                     if (it != p.hand.end()) {
                         CardInstance c = *it;
                         p.hand.erase(it);
                         c.is_tapped = false;
                         p.graveyard.push_back(c);
                         discarded_count++;
                         break; // Found and discarded
                     }
                 }
            }

            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = discarded_count;
            }
        }
    };
}
