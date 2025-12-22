from typing import List, Dict, Any, Set
from .vector_analyzer import VectorAnalyzer, CardEffectVector, ZONE_MANA, ZONE_HAND, ZONE_DECK, ZONE_BATTLE, ZONE_GRAVEYARD

class RoleClassifier:
    """
    Classifies cards into roles based on their effect vectors and stats.
    Roles:
    - INITIAL_MOVE: Low cost resource generation or hand filtering.
    - ENGINE: Mid-game advantage generation (Draw, Search, Recursion).
    - FINISHER: Win condition (High power, Breaker, Speed Attacker).
    - DEFENSE: Protection (Shield Trigger, Blocker, G-Strike).
    - META: Disruption (Discard, Cost increase, etc.).
    """

    ROLE_INITIAL_MOVE = "INITIAL_MOVE"
    ROLE_ENGINE = "ENGINE"
    ROLE_FINISHER = "FINISHER"
    ROLE_DEFENSE = "DEFENSE"
    ROLE_META = "META"
    ROLE_FLEX = "FLEX"

    def __init__(self):
        self.analyzer = VectorAnalyzer()

    def classify_card(self, card_data: Dict[str, Any]) -> Set[str]:
        roles = set()
        vectors = self.analyzer.analyze_card(card_data)

        cost = card_data.get("cost", 0)
        power = card_data.get("power", 0)
        keywords = card_data.get("keywords", {})

        # --- DEFENSE ---
        if keywords.get("shield_trigger") or keywords.get("blocker") or keywords.get("g_strike"):
            roles.add(self.ROLE_DEFENSE)

        # --- INITIAL MOVE (2-3 Cost, Advantage) ---
        if 2 <= cost <= 3:
            # Check for ramp (Any -> Mana)
            has_ramp = any(v.destination_zone == ZONE_MANA for v in vectors if not v.is_self_move)
            # Check for draw/filter (Deck -> Hand, or Hand filtering)
            has_draw = any(v.source_zone == ZONE_DECK and v.destination_zone == ZONE_HAND for v in vectors)
            has_search = any(v.source_zone == ZONE_DECK for v in vectors) # Broad search check

            if has_ramp or has_draw or has_search:
                roles.add(self.ROLE_INITIAL_MOVE)

        # --- ENGINE (Resource Extension / Loop Parts) ---
        # Any card that moves cards between zones efficiently, usually cost 4+ or repeatable
        has_advantage = any(
            (v.source_zone == ZONE_DECK and v.destination_zone == ZONE_HAND) or
            (v.source_zone == ZONE_GRAVEYARD and v.destination_zone == ZONE_HAND)
            for v in vectors
        )
        if has_advantage:
            roles.add(self.ROLE_ENGINE)

        # Revolution Change is often an Engine or Finisher
        if keywords.get("revolution_change"):
            # If it has high power/breaker, it's finisher.
            # If it draws/swaps, it's engine.
            roles.add(self.ROLE_ENGINE)

        # --- FINISHER ---
        # High Power, Multi-Breaker, Speed Attacker
        is_high_power = power >= 6000
        has_breaker = keywords.get("double_breaker") or keywords.get("triple_breaker") or keywords.get("world_breaker")
        is_sa = keywords.get("speed_attacker")

        if (is_high_power and has_breaker) or is_sa:
            roles.add(self.ROLE_FINISHER)

        # --- META ---
        # Discard, Tap, Destroy Opponent
        has_discard = any(v.destination_zone == ZONE_GRAVEYARD and v.source_zone == ZONE_HAND and "OPPONENT" in str(v.condition or "") for v in vectors)

        # Fallback: If no role assigned, it's FLEX
        if not roles:
            roles.add(self.ROLE_FLEX)

        return roles
