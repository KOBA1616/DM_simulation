#pragma once

#include "core/card_def.hpp"
#include "core/card_json_types.hpp"

namespace dm::engine::infrastructure {

    /**
     * @brief Expands specific keyword abilities into full Effect/Action definitions.
     *
     * Handles complex keywords like "Friend Burst" or "Mega Last Burst" that require
     * generating specific logic (actions/commands) rather than just setting a boolean flag.
     * This separates the expansion logic from the JSON loading process.
     */
    class KeywordExpander {
    public:
        /**
         * @brief Expands keywords from CardData into CardDefinition effects.
         *
         * @param data The raw CardData loaded from JSON.
         * @param def The CardDefinition being constructed.
         */
        static void expand_keywords(const dm::core::CardData& data, dm::core::CardDefinition& def);

    private:
        // Helpers for specific keywords
        static void expand_friend_burst(dm::core::CardDefinition& def);
        static void expand_mega_last_burst(const dm::core::CardData& data, dm::core::CardDefinition& def);
    };

}
