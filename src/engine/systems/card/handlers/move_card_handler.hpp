#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"

namespace dm::engine {

    class MoveCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
            using namespace dm::core;

            // If filter specifies zones, use generic selection
            // Otherwise, we might need to rely on source_zone string

            std::vector<int> targets;

            // If we have targets from previous selection (e.g. SELECT_TARGET)
            // But MOVE_CARD usually is the action itself that might select or operate on implicit target
            // If it has filter and no targets, it might need to select (which GenericSystem handles before calling resolve if needed?)
            // Wait, GenericSystem calls resolve directly if target scope is not TARGET_SELECT.
            // If it IS TARGET_SELECT, GenericSystem does select_targets first, then calls resolve_with_targets.

            // If scope is NONE or something else, maybe we target implicitly?

            // Let's assume we are in resolve_with_targets if targets are needed.
            // If we are here, either targets are implicit (e.g. "ALL") or already selected?
            // Actually, GenericSystem::resolve_action calls handler->resolve.

            // If the action requires selection but hasn't done it, we might be in trouble.
            // But typically ActionType::MOVE_CARD might use Scope::TARGET_SELECT.
            // In that case, GenericSystem loop sees SELECT scope and delegates to Handler?
            // No, GenericSystem::resolve_effect_with_targets calls resolve_with_targets.

            // So, resolve() is for non-targeted or self-targeted stuff.

            // Let's implement logic based on source/dest strings.

            // For now, let's look at `resolve_with_targets` mainly, and `resolve` for things like "Top of Deck".
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            const auto& action = ctx.action;
            GameState& game_state = ctx.game_state;

            if (!ctx.targets) return;

            std::string dest = action.destination_zone; // HAND, MANA_ZONE, GRAVEYARD, DECK_BOTTOM, SHIELD_ZONE

            // Helper to get zone vector
            // But instances are identified by ID. We need to find them.
            // We use GenericCardSystem::find_instance or similiar, but we need to know where they are to remove them.
            // Since we have targets list, we iterate.

            for (int target_id : *ctx.targets) {
                 move_card_to_dest(game_state, target_id, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

    private:
        void move_card_to_dest(dm::core::GameState& game_state, int instance_id, const std::string& dest, int /*source_instance_id*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;

            // 1. Find and Remove
            CardInstance card;
            bool found = false;
            bool from_battle_zone = false;
            PlayerID owner_id = 0; // Temporary default

            // We need to find the card in ANY zone of ANY player (or specific player if filtered)
            // And remove it.

            for (auto& p : game_state.players) {
                // Battle Zone
                auto b_it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (b_it != p.battle_zone.end()) {
                    card = *b_it;
                    // Hierarchy Cleanup
                    ZoneUtils::on_leave_battle_zone(game_state, *b_it);

                    p.battle_zone.erase(b_it);
                    found = true;
                    from_battle_zone = true;
                    owner_id = p.id;
                    break;
                }

                // Hand
                auto h_it = std::find_if(p.hand.begin(), p.hand.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (h_it != p.hand.end()) {
                    card = *h_it;
                    p.hand.erase(h_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Mana
                auto m_it = std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (m_it != p.mana_zone.end()) {
                    card = *m_it;
                    p.mana_zone.erase(m_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Shield
                auto s_it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (s_it != p.shield_zone.end()) {
                    card = *s_it;
                    p.shield_zone.erase(s_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Graveyard
                auto g_it = std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (g_it != p.graveyard.end()) {
                    card = *g_it;
                    p.graveyard.erase(g_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Deck?
                // Iterating deck might be slow but necessary for SEARCH.
                // Usually SEARCH handler removes it. But if MOVE_CARD targets DECK...
                // Let's check deck.
                 auto d_it = std::find_if(p.deck.begin(), p.deck.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (d_it != p.deck.end()) {
                    card = *d_it;
                    p.deck.erase(d_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }
            }

            if (!found) {
                // Check buffers (Player 0)
                auto buf_it_p0 = std::find_if(game_state.players[0].effect_buffer.begin(), game_state.players[0].effect_buffer.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (buf_it_p0 != game_state.players[0].effect_buffer.end()) {
                    card = *buf_it_p0;
                    game_state.players[0].effect_buffer.erase(buf_it_p0);
                    found = true;
                    owner_id = 0; // Found in P0's buffer
                } else {
                    // Check buffers (Player 1)
                    auto buf_it_p1 = std::find_if(game_state.players[1].effect_buffer.begin(), game_state.players[1].effect_buffer.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                    if (buf_it_p1 != game_state.players[1].effect_buffer.end()) {
                        card = *buf_it_p1;
                        game_state.players[1].effect_buffer.erase(buf_it_p1);
                        found = true;
                        owner_id = 1; // Found in P1's buffer
                    }
                }
            }

            if (!found) return;

            // 2. Add to Destination
            // Use owner_id? Or target_player of action?
            // Usually cards return to OWNER's zone.
            Player& owner = game_state.players[owner_id];

            if (dest == "HAND") {
                // Reset state
                card.is_tapped = false;
                card.summoning_sickness = true;
                owner.hand.push_back(card);
                if (from_battle_zone) {
                    GenericCardSystem::check_mega_last_burst(game_state, card, card_db);
                }
            } else if (dest == "MANA_ZONE") {
                card.is_tapped = false; // Usually untapped unless specified?
                // "Put into mana zone" -> usually untapped. "Charge" -> untapped.
                // Some effects put tapped. But default to untapped.
                owner.mana_zone.push_back(card);
                if (from_battle_zone) {
                    GenericCardSystem::check_mega_last_burst(game_state, card, card_db);
                }
            } else if (dest == "GRAVEYARD") {
                card.is_tapped = false;
                owner.graveyard.push_back(card);
                if (from_battle_zone) {
                    GenericCardSystem::check_mega_last_burst(game_state, card, card_db);
                }
            } else if (dest == "DECK_BOTTOM") {
                card.is_tapped = false;
                owner.deck.insert(owner.deck.begin(), card);
            } else if (dest == "DECK_TOP") {
                card.is_tapped = false;
                owner.deck.push_back(card);
            } else if (dest == "SHIELD_ZONE") {
                card.is_tapped = false;
                owner.shield_zone.push_back(card);
                GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_SHIELD_ADD, card.instance_id, card_db);
            } else if (dest == "BATTLE_ZONE") {
                card.is_tapped = false;
                card.summoning_sickness = true;
                card.turn_played = game_state.turn_number;
                owner.battle_zone.push_back(card);
                // ON_PLAY?
            }
        }
    };
}
