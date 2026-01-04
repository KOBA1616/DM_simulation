#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <map>

namespace dm::engine::systems {

    class TriggerSystem {
    public:
        static TriggerSystem& instance() {
            static TriggerSystem instance;
            return instance;
        }

        // Queues pending effects for the specified trigger on the source card
        void resolve_trigger(core::GameState& game_state, core::TriggerType trigger, int source_instance_id, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Helper to queue a generic pending effect
        void add_pending_effect(core::GameState& game_state, const core::PendingEffect& pending_effect);

    private:
        TriggerSystem() = default;
        TriggerSystem(const TriggerSystem&) = delete;
        TriggerSystem& operator=(const TriggerSystem&) = delete;

        core::PlayerID get_controller(const core::GameState& game_state, int instance_id);
    };
}
