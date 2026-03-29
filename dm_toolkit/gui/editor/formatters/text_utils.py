from typing import List, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources

class TextUtils:
    """Utility class for handling common Japanese punctuation and verb conjugations."""

    @staticmethod
    def format_comparison_operator(op: str, value: Any) -> str:
        """
        Formats a comparison operator and value into Japanese text.
        """
        if op == ">=":
            return f"{value}以上"
        elif op == "<=":
            return f"{value}以下"
        elif op in ("=", "=="):
            return f"{value}"
        elif op == ">":
            return f"{value}より多い"
        elif op == "<":
            return f"{value}未満"
        return f"{value}"

    @staticmethod
    def apply_conjugation(text: str, optional: bool = False) -> str:
        """
        Applies standard conjugation to the end of a sentence based on the `optional` flag.
        Handles replacing '。' with 'てもよい。' etc. using data-driven rules.
        Instead of merely changing the string ending, this splits by '。' and applies
        conjugation only to the final non-empty clause, ensuring multiple or nested clauses
        are handled safely at a structural level.
        """
        if not optional or not text:
            return text

        # Split into distinct sentences/clauses
        sentences = text.split("。")

        # Usually sentences split by "。" will have an empty string at the very end if text ends with "。"
        # We need to find the last substantive sentence
        last_idx = -1
        for i in range(len(sentences) - 1, -1, -1):
            if sentences[i].strip():
                last_idx = i
                break

        if last_idx == -1:
            return text

        target_sentence = sentences[last_idx] + "。"
        rules = CardTextResources.CONJUGATION_RULES

        conjugated = target_sentence
        applied = False

        for suffix, conjugated_suffix in rules.items():
            if target_sentence.endswith(suffix):
                conjugated = target_sentence[:-len(suffix)] + conjugated_suffix
                applied = True
                break

        if not applied and not target_sentence.endswith("てもよい。"):
            conjugated = target_sentence[:-1] + "てもよい。"

        # Re-attach the conjugated sentence
        sentences[last_idx] = conjugated[:-1] # Remove the trailing "。" for joining later

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
