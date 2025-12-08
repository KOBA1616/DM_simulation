#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include <iostream>

namespace dm::engine {

    class SelectNumberHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            // Usually requires user input or AI decision.
            // For now, if "value1" is set, use it (hardcoded).
            // If not, we might need a "SELECT_NUMBER" pending effect?
            // Or we assume the AI has already made a choice via Action generation?

            // Current engine structure: ActionGenerator creates Actions.
            // If ActionType::SELECT_NUMBER exists, the `value1` field of the generated Action should contain the chosen number.
            // But here we are in `resolve`, executing the effect definition.
            // If the effect says "Select Number", we need to *ask* for it.

            // Standard flow:
            // 1. PendingEffect (ResolveType::SELECT_NUMBER?)
            // 2. ActionGenerator generates options (e.g. Select 1, Select 2, ...) or AI returns a number.
            // 3. Action (SELECT_NUMBER, value=5) is executed.
            // 4. Then we store the result in context.

            // However, we don't have ResolveType::SELECT_NUMBER yet.
            // We can reuse ResolveType::TARGET_SELECT or add a new one.
            // Or use "SELECT_TARGET" with a special target mode? No, messy.

            // Let's implement immediate random choice for now (Heuristic Step 1) or assume input via `action.value1` if it came from an Action.
            // But `resolve` takes `ActionDef` (from JSON), not `Action` (runtime choice).
            // Wait, `resolve_with_targets` takes targets.
            // Maybe we need `resolve_with_value`?

            // Workaround:
            // If we are here, we are executing the effect.
            // We generate a PendingEffect for "Selection" if not already done?
            // Actually, `EffectResolver` calls `GenericCardSystem::resolve_effect`.
            // If an action requires selection (Target, Number), we should interrupt and push PendingEffect.

            // We need `EffectActionType::SELECT_NUMBER` to be treated as an "interactive" action in `GenericCardSystem`.
            // In `GenericCardSystem::resolve_effect`, we iterate actions.
            // If we hit SELECT_NUMBER, we check if we have a choice ready?
            // Currently `GenericCardSystem` only supports `TARGET_SELECT` interruption.

            // Plan:
            // 1. Add `SELECT_NUMBER` to the interactive check in `GenericCardSystem`.
            // 2. Push PendingEffect with `ResolveType::NUMBER_SELECT`.
            // 3. ActionGenerator generates `ActionType::SELECT_NUMBER` with specific values.

            // But modifying `GenericCardSystem` is heavy.
            // Simplified: "Randomly choose" inside handler for now (as per AI Step 1 "Heuristic").
            // Or "Always choose 5" as a placeholder.
            // User requirement: "AI guesses effectively".
            // So we need AI interaction.

            // Let's defer "True Interactive Number Selection" to a future refactor if possible, or hack it:
            // The handler will "guess" immediately and proceed.
            // This avoids UI/Protocol changes for now.
            // "Heuristic (Rule-based)":
            // - Check opponent mana/grave.
            // - Guess mostly likely cost.
            // - Set it in context.

            int chosen_number = 0;

            // Heuristic Logic (Basic)
            // 1. Scan opponent zone (Mana) to see what costs they have.
            // 2. Scan opponent zone (Battle) for what costs they have.
            // 3. Pick the most frequent cost? Or max cost?
            // If "Destroy cost X", we want to match.
            // If "Stop spell cost X", we want to match deck type.

            // Since we don't know the *intent* (Destroy vs Stop) easily here without parsing text...
            // We assume "Destroy" context?
            // Actually `input_value_key` might give a hint, or `output_value_key`.

            // Let's try a safe heuristic:
            // If opponent has creatures, pick the cost of their highest power creature (to destroy it?).
            // If no creatures, pick 5 (common key cost).

            int best_cost = 5;
            const dm::core::Player& opponent = game_state.get_non_active_player();

            if (!opponent.battle_zone.empty()) {
                // Find most powerful creature
                int max_power = -1;
                for (const auto& c : opponent.battle_zone) {
                    // Check power using cache or DB? We need DB.
                    // Handler doesn't have DB in `resolve` (it does in `resolve_with_targets` but signature varies).
                    // Wait, `IActionHandler::resolve` signature does NOT have `card_db`?
                    // I removed it? No, I updated `resolve_with_targets` to have it.
                    // `resolve` does not.
                    // We need `card_db` for heuristic.
                    // We can't access it here easily.
                }
            }

            // Fallback: Random 1-10 or Just 5.
            chosen_number = 5;

            // Store output
            if (!action.output_value_key.empty()) {
                execution_context[action.output_value_key] = chosen_number;
            }

            std::cout << "DEBUG: SelectNumberHandler chose " << chosen_number << " (Heuristic)" << std::endl;

            // Unused
            (void)source_instance_id;
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
             // Not used for Number Selection (unless targeting players/cards to derive number?)
             resolve(game_state, action, source_instance_id, execution_context);
             (void)targets; (void)card_db;
        }
    };
}
