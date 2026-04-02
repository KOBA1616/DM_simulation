from typing import List, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources

class TextUtils:
    """Utility class for handling common Japanese punctuation and verb conjugations."""

    @staticmethod
    def format_comparison_operator(op: str, value: Any, attribute: str = "", particle: str = "の", attribute_type: str = "") -> str:
        """
        Formats a comparison operator and value into Japanese text.
        If an attribute is provided, generates phrases like "X以下のコストの〜".
        `attribute_type` can be specified (e.g., 'cost', 'count', 'power') to drive context-specific phrasing.
        """
        # Infer attribute_type from attribute string if not explicitly given
        attr_str = str(attribute)
        if not attribute_type:
             if "コスト" in attr_str: attribute_type = "cost"
             elif "パワー" in attr_str: attribute_type = "power"
             elif "数" in attr_str or "枚" in attr_str or "体" in attr_str: attribute_type = "count"

        op_text = ""
        if op == ">=":
            op_text = f"{value}以上"
        elif op == "<=":
            op_text = f"{value}以下"
        elif op in ("=", "=="):
            op_text = f"{value}"
        elif op == ">":
            if attribute_type == "power":
                op_text = f"{value}より大きい"
            else:
                op_text = f"{value}より多い"
        elif op == "<":
            if attribute_type in ["cost", "count"]:
                 op_text = f"{value}より少ない"
            else:
                 op_text = f"{value}未満"
        else:
            op_text = f"{value}"

        if not attribute:
            return op_text

        if op in ("=", "=="):
            return f"{attribute}{op_text}{particle}"
        elif op == ">" and attribute_type == "power":
            # Avoid appending "の" after an i-adjective like "大きい"
            adj_particle = "" if particle == "の" else particle
            return f"{attribute}が{op_text}{adj_particle}"
        elif op == "<" and attribute_type in ["cost", "count"]:
            adj_particle = "" if particle == "の" else particle
            return f"{attribute}が{op_text}{adj_particle}"

        return f"{op_text}の{attribute}{particle}"

    @staticmethod
    def apply_conjugation(text: str, optional: bool = False, is_composite_action: bool = False) -> str:
        """
        Applies standard conjugation to the end of a sentence based on the `optional` flag.
        Handles replacing '。' with 'てもよい。' etc. using data-driven rules.
        Instead of merely changing the string ending, this splits by '。' and applies
        conjugation only to the final non-empty clause, ensuring multiple or nested clauses
        are handled safely at a structural level.
        `is_composite_action` allows context mergers or formatters to specify that the
        text being conjugated is part of a larger, grouped clause structure where normal end-of-sentence
        conjugation should be carefully evaluated.
        """
        if not optional or not text:
            return text

        # Split into distinct sentences/clauses
        sentences = text.split("。")

        # Usually sentences split by "。" will have an empty string at the very end if text ends with "。"
        # We need to find the last substantive sentence that should be conjugated.
        # In DM, composite sentences like "〜する。その後、〜する。" often shouldn't have the final "その後" conjugated
        # IF the "その後" is a mandatory secondary action. However, since we process command by command,
        # ContextMerger might group them into a single block "〜する。その後、〜する。"
        # If the user sets `optional: True` on the primary command but ContextMerger glued them, we might incorrectly conjugate the end.
        # To avoid this, we will find the FIRST sentence before "その後" or just use AST/CommandListFormatter.
        # Wait, the simplest fix for "複合文において意図しない変換が起きるリスク" is not splitting blindly if the sequence implies a single conjugated block.
        # Actually, let's fix the target sentence finding: if we have "その後、" we might want to conjugate the sentence *before* it.
        last_idx = -1
        for i in range(len(sentences) - 1, -1, -1):
            if sentences[i].strip():
                last_idx = i
                break

        if last_idx == -1:
            return text

        # Context-aware composite clause safety check
        # If this action is part of a composite sequence and the final segment starts with "その後、",
        # the optionality ("〜てもよい") typically applies to the primary action (the preceding segment)
        # rather than the follow-up action. We backtrack to conjugate the principal sentence.
        if is_composite_action and sentences[last_idx].strip().startswith("その後、"):
            # Move backward to find the previous non-empty sentence
            for i in range(last_idx - 1, -1, -1):
                if sentences[i].strip():
                    last_idx = i
                    break

        target_sentence = sentences[last_idx] + "。"

        # We augment CardTextResources.CONJUGATION_RULES with missing Godan verb forms to be robust
        # This replaces the hardcoded "る" hack with standard conjugation rules
        base_rules = CardTextResources.CONJUGATION_RULES

        # Extend with common verbs missing from base resources
        rules = {
            "する。": "してもよい。",
            "く。": "いてもよい。",
            "ぐ。": "いでもよい。",
            "す。": "してもよい。",
            "つ。": "ってもよい。",
            "ぬ。": "んでもよい。",
            "ぶ。": "んでもよい。",
            "む。": "んでもよい。",
            "る。": "ってもよい。",  # Fixes Godan verbs like "めくる" -> "めくってもよい"
            "う。": "ってもよい。"
        }

        # Some special Ichidan verbs might override "る。" rule. In DM text,
        # "捨てる" -> "捨ててもよい", "立てる" -> "立ててもよい".
        # If it ends in "える" or "いる", it is likely Ichidan.
        # We can handle this procedurally.

        conjugated = target_sentence
        applied = False

        if target_sentence.endswith("る。"):
             # Check if it's an Ichidan verb ending (e.g. える / いる sounds)
             # DM text mostly uses: 捨てる、出させる、加える、離れる、いる
             ichidan_endings = ["える。", "いる。", "れる。", "せる。", "てる。"]
             is_ichidan = any(target_sentence.endswith(e) for e in ichidan_endings)
             if is_ichidan:
                 conjugated = target_sentence[:-2] + "てもよい。"
                 applied = True
             else:
                 # Godan verb like "めくる", "乗る"
                 conjugated = target_sentence[:-2] + "ってもよい。"
                 applied = True

        if not applied:
            for suffix, conjugated_suffix in rules.items():
                if target_sentence.endswith(suffix):
                    conjugated = target_sentence[:-len(suffix)] + conjugated_suffix
                    applied = True
                    break

        if not applied:
             if not target_sentence.endswith("てもよい。"):
                 conjugated = target_sentence[:-1] + "てもよい。"

        # Re-attach the conjugated sentence
        sentences[last_idx] = conjugated[:-1] # Remove the trailing "。" for joining later

        # Ensure composite sentence joins remain fully correct
        return "。".join(sentences)

    @staticmethod
    def join_sentences(sentences: List[str]) -> str:
        """
        Joins a list of Japanese sentences cleanly.
        """
        if not sentences:
            return ""

        filtered = [s for s in sentences if s]
        if not filtered:
            return ""

        return "。".join([s.rstrip("。") for s in filtered]) + "。"

    @staticmethod
    def format_up_to(amount: Any, unit: str, up_to: bool) -> str:
        """
        Generates the proper Japanese phrase for a quantity, factoring in 'up to' logic.
        E.g., amount=2, unit="枚", up_to=True -> "最大2枚まで"
              amount=2, unit="枚", up_to=False -> "2枚"
        """
        if up_to:
            return f"最大{amount}{unit}まで"
        return f"{amount}{unit}"
