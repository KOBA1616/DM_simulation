#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dm::engine {

    struct ResolutionContext {
        dm::core::GameState& game_state;
        const dm::core::ActionDef& action;
        int source_instance_id;
        std::map<std::string, int>& execution_vars;
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db;
        const std::vector<int>* targets = nullptr;

        ResolutionContext(
            dm::core::GameState& state,
            const dm::core::ActionDef& act,
            int src,
            std::map<std::string, int>& vars,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& db,
            const std::vector<int>* tgs = nullptr)
            : game_state(state), action(act), source_instance_id(src),
              execution_vars(vars), card_db(db), targets(tgs) {}
    };

    class IActionHandler {
    public:
        virtual ~IActionHandler() = default;
        virtual void resolve(const ResolutionContext& ctx) = 0;
        virtual void resolve_with_targets([[maybe_unused]] const ResolutionContext& ctx) {}
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
