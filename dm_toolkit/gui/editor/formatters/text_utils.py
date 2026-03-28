from typing import List
from dm_toolkit.gui.editor.text_resources import CardTextResources

class TextUtils:
    """Utility class for handling common Japanese punctuation and verb conjugations."""

    @staticmethod
    def apply_conjugation(text: str, optional: bool = False) -> str:
        """
        Applies standard conjugation to the end of a sentence based on the `optional` flag.
        Handles replacing '。' with 'てもよい。' etc. using data-driven rules.
        """
        if not optional:
            return text

        rules = CardTextResources.CONJUGATION_RULES

        for suffix, conjugated_suffix in rules.items():
            if text.endswith(suffix):
                return text[:-len(suffix)] + conjugated_suffix

        if not text.endswith("てもよい。"):
            return text[:-1] + "てもよい。"

        return text

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
