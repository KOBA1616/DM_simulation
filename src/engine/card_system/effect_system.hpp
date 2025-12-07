#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dm::engine {

    class IActionHandler {
    public:
        virtual ~IActionHandler() = default;
        virtual void resolve(dm::core::GameState& state, const dm::core::ActionDef& action, int source_id, std::map<std::string, int>& context) = 0;
        virtual void resolve_with_targets(dm::core::GameState& state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context) {}
    };

    class EffectSystem {
    public:
        static EffectSystem& instance() {
            static EffectSystem instance;
            return instance;
        }

        void register_handler(dm::core::EffectActionType type, std::unique_ptr<IActionHandler> handler) {
            handlers[type] = std::move(handler);
        }

        IActionHandler* get_handler(dm::core::EffectActionType type) {
            if (handlers.count(type)) {
                return handlers[type].get();
            }
            return nullptr;
        }

    private:
        EffectSystem() = default;
        std::map<dm::core::EffectActionType, std::unique_ptr<IActionHandler>> handlers;
    };
}
