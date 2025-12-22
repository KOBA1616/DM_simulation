from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

# Zone Constants
ZONE_DECK = "DECK"
ZONE_HAND = "HAND"
ZONE_MANA = "MANA_ZONE"
ZONE_GRAVEYARD = "GRAVEYARD"
ZONE_BATTLE = "BATTLE_ZONE"
ZONE_SHIELD = "SHIELD_ZONE"
ZONE_NONE = "NONE"

# Special destinations for search/look
ZONE_DECK_BOTTOM = "DECK_BOTTOM"

@dataclass
class CardEffectVector:
    """
    Represents a directed movement of cards between zones.
    """
    source_zone: str
    destination_zone: str
    count: int  # 0 usually means 'amount' from action, or variable
    target_filter: Optional[Dict[str, Any]] = None
    is_self_move: bool = False  # True if the card itself is moving (e.g. Summon)
    condition: Optional[str] = None

    def __str__(self):
        arrow = "=>" if self.is_self_move else "->"
        return f"[{self.source_zone} {arrow} {self.destination_zone}] (x{self.count})"

class VectorAnalyzer:
    """
    Analyzes card definitions to extract effect vectors.
    """

    def __init__(self):
        self.logger = logging.getLogger("VectorAnalyzer")

    def analyze_card(self, card_data: Dict[str, Any]) -> List[CardEffectVector]:
        vectors = []

        # 1. Implicit Self-Movement (Playing the card)
        card_type = card_data.get("type", "CREATURE")
        if card_type == "CREATURE":
            vectors.append(CardEffectVector(ZONE_HAND, ZONE_BATTLE, 1, is_self_move=True))
        elif card_type == "SPELL":
            # Spells go to GY after resolving.
            vectors.append(CardEffectVector(ZONE_HAND, ZONE_GRAVEYARD, 1, is_self_move=True))

        # NOTE: Removed keyword-based Generic Vector for Revolution Change to avoid duplicates,
        # as the specific ACTIONS (REVOLUTION_CHANGE) in effects usually cover this.

        # 2. Effects
        for effect in card_data.get("effects", []):
            vectors.extend(self._analyze_effect(effect))

        # 3. Triggers (Some cards have triggers outside effects list or in separate triggers list)
        for trigger in card_data.get("triggers", []):
            vectors.extend(self._analyze_effect(trigger))

        return vectors

    def _analyze_effect(self, effect: Dict[str, Any]) -> List[CardEffectVector]:
        vectors = []

        # Handle 'actions' (New style)
        if "actions" in effect:
            for action in effect["actions"]:
                vectors.extend(self._analyze_action(action))

        # Handle 'commands' (Legacy style)
        if "commands" in effect:
            for command in effect["commands"]:
                vectors.extend(self._analyze_command(command))

        return vectors

    def _analyze_action(self, action: Dict[str, Any]) -> List[CardEffectVector]:
        vectors = []
        action_type = action.get("type")

        if action_type == "DRAW_CARD":
            count = action.get("value1", 1)
            vectors.append(CardEffectVector(ZONE_DECK, ZONE_HAND, count))

        elif action_type == "MANA_CHARGE":
             vectors.append(CardEffectVector(ZONE_HAND, ZONE_MANA, 1))

        elif action_type == "PLAY_FROM_ZONE":
            source = ZONE_HAND
            dest = ZONE_BATTLE

            # Check filter for source
            zones = action.get("filter", {}).get("zones", [])
            if zones:
                source = zones[0]

            # Check filter for card type to determine destination (Spell -> Grave)
            types = action.get("filter", {}).get("types", [])
            if "SPELL" in types:
                dest = ZONE_GRAVEYARD

            vectors.append(CardEffectVector(source, dest, 1))

        elif action_type == "SEARCH_DECK":
             vectors.append(CardEffectVector(ZONE_DECK, ZONE_HAND, 1))

        elif action_type == "FRIEND_BURST":
             # Cost: Battle -> Hand (Self/Friendly)
             vectors.append(CardEffectVector(ZONE_BATTLE, ZONE_HAND, 1))

        elif action_type == "REVOLUTION_CHANGE":
             # "destination_zone": "HAND" -> This is for the creature returning to hand.
             # This action represents the 'Swap' cost part (The creature returning).
             # The 'Self' entering is handled by the implicit Self-Movement of the new card.
             vectors.append(CardEffectVector(ZONE_BATTLE, ZONE_HAND, 1))

        elif action_type == "TRANSITION":
             pass # Logic handled by _analyze_command usually, but if it appears here...
             # Actions usually don't use 'TRANSITION' type, they use MOVE_CARD or specific types.

        return vectors

    def _analyze_command(self, command: Dict[str, Any]) -> List[CardEffectVector]:
        vectors = []
        cmd_type = command.get("type")

        if cmd_type == "TRANSITION":
            source = command.get("from_zone", ZONE_NONE)
            dest = command.get("to_zone", ZONE_NONE)
            count = command.get("amount", 1)

            if source != ZONE_NONE and dest != ZONE_NONE:
                 vectors.append(CardEffectVector(source, dest, count))

        elif cmd_type == "DRAW_CARD":
            count = command.get("amount", 1)
            vectors.append(CardEffectVector(ZONE_DECK, ZONE_HAND, count))

        elif cmd_type == "DISCARD":
            count = command.get("amount", 1)
            vectors.append(CardEffectVector(ZONE_HAND, ZONE_GRAVEYARD, count))

        return vectors
