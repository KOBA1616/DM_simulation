#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dm::engine::effects {

    class EffectSystem {
    public:
        static EffectSystem& instance() {
            static EffectSystem instance;
            return instance;
        }

        void initialize();

        // CommandDef-based effect resolution
        void resolve_effect(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        void resolve_effect_with_context(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        void resolve_effect_with_targets(dm::core::GameState& game_state, const dm::core::EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context);

        bool check_condition(dm::core::GameState& game_state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context = {});

        static dm::core::PlayerID get_controller(const dm::core::GameState& game_state, int instance_id);

    private:
        EffectSystem() = default;
        EffectSystem(const EffectSystem&) = delete;
        EffectSystem& operator=(const EffectSystem&) = delete;
        bool initialized = false;
    };
}
