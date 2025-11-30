#include "generic_card_system.hpp"
#include "card_registry.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    // Helper to find card instance
    static CardInstance* find_instance(GameState& game_state, int instance_id) {
        for (auto& p : game_state.players) {
            for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
        }
        return nullptr;
    }

    void GenericCardSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id) {
        CardInstance* instance = find_instance(game_state, source_instance_id);
        if (!instance) return;

        const CardData* data = CardRegistry::get_card_data(instance->card_id);
        if (!data) return;

        for (const auto& effect : data->effects) {
            if (effect.trigger == trigger) {
                resolve_effect(game_state, effect, source_instance_id);
            }
        }
    }

    void GenericCardSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id) {
        if (!check_condition(game_state, effect.condition, source_instance_id)) return;

        for (const auto& action : effect.actions) {
            resolve_action(game_state, action, source_instance_id);
        }
    }

    bool GenericCardSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id) {
        if (condition.type == "NONE") return true;
        // Implement other conditions like MANA_ARMED
        return true;
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        // Handle simple actions immediately
        // Handle complex actions (Target Selection) by pushing PendingEffect?
        // For now, let's implement simple auto-target actions.

        Player& active = game_state.get_active_player();
        // Player& opponent = game_state.get_non_active_player();

        switch (action.type) {
            case ActionType::DRAW_CARD: {
                int count = action.value1;
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) {
                        game_state.winner = (active.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                        return;
                    }
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    active.hand.push_back(c);
                }
                break;
            }
            case ActionType::ADD_MANA: {
                int count = action.value1;
                for (int i = 0; i < count; ++i) {
                    if (active.deck.empty()) break;
                    CardInstance c = active.deck.back();
                    active.deck.pop_back();
                    c.is_tapped = false; // Usually untapped
                    active.mana_zone.push_back(c);
                }
                break;
            }
            case ActionType::DESTROY: {
                // If scope is TARGET_SELECT, we need to handle it.
                // If scope is ALL_FILTERED, we do it now.
                // For now, just log.
                // std::cout << "Destroy action triggered" << std::endl;
                break;
            }
            default:
                break;
        }
    }

    std::vector<int> GenericCardSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id) {
        return {};
    }

}
