#pragma once
#include "types.hpp"
#include "card_json_types.hpp"
#include <vector>
#include <optional>
#include <map>
#include <string>

namespace dm::core {

    struct PendingEffect {
        EffectType type;
        int source_instance_id;
        PlayerID controller;

        // Targeting
        std::vector<int> target_instance_ids;
        int num_targets_needed = 0;
        ResolveType resolve_type = ResolveType::NONE;

        FilterDef filter; // The filter used for selection
        bool optional = false; // If true, can choose to select nothing (PASS)

        // Optional: carry the EffectDef (from JSON) for later resolution after target selection
        std::optional<EffectDef> effect_def;

        // For SELECT_OPTION: Store the choices
        std::vector<std::vector<ActionDef>> options;

        // Phase 5: Execution Context (Variable Linking)
        std::map<std::string, int> execution_context;

        // Step 5.2.2: Loop Prevention
        int chain_depth = 0;

        // Optional context for REACTION_WINDOW
        struct ReactionContext {
            std::string trigger_event; // The event being reacted to (e.g., "ON_BLOCK_OR_ATTACK")
            int attacking_creature_id = -1; // Instance ID of attacker
            int blocked_creature_id = -1;
        };
        std::optional<ReactionContext> reaction_context;

        PendingEffect(EffectType t, int src, PlayerID p)
            : type(t), source_instance_id(src), controller(p) {}
    };

}
