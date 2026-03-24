#include "keyword_expander.hpp"
#include <iostream>

namespace dm::engine::infrastructure {

    using namespace dm::core;

    void dm::engine::infrastructure::KeywordExpander::expand_keywords(const CardData& data, CardDefinition& def) {
        if (!data.keywords.has_value()) {
            return;
        }

        const auto& kws = *data.keywords;

        // Friend Burst
        if (kws.count("friend_burst") && kws.at("friend_burst")) {
            expand_friend_burst(def);
        }

        // Mega Last Burst
        if (kws.count("mega_last_burst") && kws.at("mega_last_burst")) {
            expand_mega_last_burst(data, def);
        }
    }

    void dm::engine::infrastructure::KeywordExpander::expand_friend_burst(CardDefinition& def) {
        def.keywords.friend_burst = true;

        EffectDef fb_effect;
        fb_effect.trigger = TriggerType::ON_PLAY;

            CommandDef fb_action;
            fb_action.type = CommandType::FRIEND_BURST;
            fb_action.target_group = TargetScope::TARGET_SELECT;
            fb_action.optional = true;
            fb_action.target_filter.owner = "SELF";
            fb_action.target_filter.zones = {"BATTLE_ZONE"};
            fb_action.target_filter.types = {"CREATURE"};
            fb_action.target_filter.is_tapped = false;
            fb_action.target_filter.count = 1;

        // Push as Command into commands list
        fb_effect.commands.push_back(fb_action);

        def.effects.push_back(fb_effect);
    }

    void dm::engine::infrastructure::KeywordExpander::expand_mega_last_burst(const CardData& data, CardDefinition& def) {
        def.keywords.mega_last_burst = true;

        if (data.spell_side) {
            EffectDef mlb_effect;
            mlb_effect.trigger = TriggerType::ON_DESTROY;

                CommandDef mlb_action;
                mlb_action.type = CommandType::CAST_SPELL;
                mlb_action.target_group = TargetScope::SELF; // Use SELF to target the card itself (in graveyard)
                mlb_action.optional = true;
                // CommandDef currently does not have cast_spell_side field; use str_val as hint
                mlb_action.str_val = "cast_spell_side";

            // Push as Command into commands list
            mlb_effect.commands.push_back(mlb_action);

            def.effects.push_back(mlb_effect);
        }
    }

}
