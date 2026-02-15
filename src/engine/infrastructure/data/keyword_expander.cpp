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

        ActionDef fb_action;
        fb_action.type = EffectPrimitive::FRIEND_BURST;
        fb_action.scope = TargetScope::TARGET_SELECT;
        fb_action.optional = true;
        fb_action.filter.owner = "SELF";
        fb_action.filter.zones = {"BATTLE_ZONE"};
        fb_action.filter.types = {"CREATURE"};
        fb_action.filter.is_tapped = false;
        fb_action.filter.count = 1;

        fb_effect.actions.push_back(fb_action);

        // Add Command counterpart for Friend Burst
        CommandDef fb_cmd;
        fb_cmd.type = CommandType::FRIEND_BURST;
        fb_cmd.target_group = TargetScope::TARGET_SELECT;
        fb_cmd.optional = true;
        fb_cmd.target_filter = fb_action.filter;
        fb_effect.commands.push_back(fb_cmd);

        def.effects.push_back(fb_effect);
    }

    void dm::engine::infrastructure::KeywordExpander::expand_mega_last_burst(const CardData& data, CardDefinition& def) {
        def.keywords.mega_last_burst = true;

        if (data.spell_side) {
            EffectDef mlb_effect;
            mlb_effect.trigger = TriggerType::ON_DESTROY;

            ActionDef mlb_action;
            mlb_action.type = EffectPrimitive::CAST_SPELL;
            mlb_action.scope = TargetScope::SELF; // Use SELF to target the card itself (in graveyard)
            mlb_action.optional = true;
            mlb_action.cast_spell_side = true;

            mlb_effect.actions.push_back(mlb_action);

            // Add Command counterpart
            CommandDef mlb_cmd;
            mlb_cmd.type = CommandType::CAST_SPELL;
            // Command Logic for MLB casting from Graveyard
            // This assumes the command interpreter knows how to handle CAST_SPELL with cast_spell_side context
            // possibly derived from the action conversion or explicitly set here if command def supports it.
            // For now, mirroring what was in dm::engine::infrastructure::JsonLoader.

            mlb_effect.commands.push_back(mlb_cmd);

            def.effects.push_back(mlb_effect);
        }
    }

}
