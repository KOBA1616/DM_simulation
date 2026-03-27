from typing import List

class TextUtils:
    """Utility class for handling common Japanese punctuation and verb conjugations."""

    @staticmethod
    def apply_conjugation(text: str, optional: bool = False) -> str:
        """
        Applies standard conjugation to the end of a sentence based on the `optional` flag.
        Handles replacing '。' with 'てもよい。' etc.
        """
        if not optional:
            return text

        if text.endswith("する。"):
            return text[:-3] + "してもよい。"
        elif text.endswith("く。"):
            return text[:-2] + "いてもよい。"
        elif text.endswith("す。"):
            return text[:-2] + "してもよい。"
        elif text.endswith("る。"):
            return text[:-2] + "てもよい。"
        elif text.endswith("う。"):
            return text[:-2] + "ってもよい。"
        elif not text.endswith("てもよい。"):
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
