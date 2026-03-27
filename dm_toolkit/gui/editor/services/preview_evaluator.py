from typing import Any, List

class PreviewEvaluator:
    @classmethod
    def compute_stat_from_sample(cls, key: str, sample: List[Any]) -> Any:
        if not sample:
            return None

        if key == "MANA_CIVILIZATION_COUNT":
            civs = set()
            for entry in sample:
                if isinstance(entry, str):
                    civs.add(entry)
                elif isinstance(entry, dict):
                    for c in entry.get('civilizations', []):
                        civs.add(c)
            return len(civs)

        count_stats = {
            "MANA_COUNT", "CREATURE_COUNT", "SHIELD_COUNT", "HAND_COUNT",
            "GRAVEYARD_COUNT", "BATTLE_ZONE_COUNT", "OPPONENT_MANA_COUNT",
            "OPPONENT_CREATURE_COUNT", "OPPONENT_SHIELD_COUNT", "OPPONENT_HAND_COUNT",
            "OPPONENT_GRAVEYARD_COUNT", "OPPONENT_BATTLE_ZONE_COUNT", "CARDS_DRAWN_THIS_TURN"
        }
        if key in count_stats:
            return len(sample)

        return None
