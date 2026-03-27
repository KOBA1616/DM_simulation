from typing import Dict, Any

class DataMigration:
    @classmethod
    def normalize_cost_reduction_dict(cls, cr: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize cost reduction dictionary to consistently use the new format.

        Extracts values from legacy 'unit_cost' field into top-level properties
        so that downstream rendering only needs to deal with one schema.
        """
        if not cr:
            return cr

        normalized = cr.copy()

        # Merge properties from legacy unit_cost if present
        unit_cost = normalized.get("unit_cost", {})
        if isinstance(unit_cost, dict):
            if "value_mode" not in normalized and "value_mode" in unit_cost:
                normalized["value_mode"] = unit_cost.get("value_mode")
            if "filter" not in normalized and "filter" in unit_cost:
                normalized["filter"] = unit_cost.get("filter")
            if "value" not in normalized and "value" in unit_cost:
                normalized["value"] = unit_cost.get("value")
            if "per_value" not in normalized and "per_value" in unit_cost:
                normalized["per_value"] = unit_cost.get("per_value")
            if "increment_cost" not in normalized and "increment_cost" in unit_cost:
                normalized["increment_cost"] = unit_cost.get("increment_cost")
            if "max_reduction" not in normalized and "max_reduction" in unit_cost:
                normalized["max_reduction"] = unit_cost.get("max_reduction")
            if "stat_key" not in normalized and "stat_key" in unit_cost:
                normalized["stat_key"] = unit_cost.get("stat_key")
            if "min_stat" not in normalized and "min_stat" in unit_cost:
                normalized["min_stat"] = unit_cost.get("min_stat")

        # Resolve common alias fields
        if "value_mode" not in normalized and "mode" in normalized:
            normalized["value_mode"] = normalized.get("mode")
        if "value" not in normalized and "amount" in normalized:
            normalized["value"] = normalized.get("amount")
        if "per_value" not in normalized and "per" in normalized:
            normalized["per_value"] = normalized.get("per")
        if "increment_cost" not in normalized and "increment" in normalized:
            normalized["increment_cost"] = normalized.get("increment")

        return normalized
