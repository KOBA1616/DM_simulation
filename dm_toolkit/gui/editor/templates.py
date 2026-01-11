# -*- coding: utf-8 -*-
import json
import os
import copy
import uuid

class LogicTemplateManager:
    """
    Manages loading and applying logic templates for the Card Editor.
    Replaces hardcoded logic construction with data-driven templates.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        # Paths to search
        search_paths = [
            os.path.join(os.getcwd(), 'data', 'editor_templates.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'editor_templates.json'),
        ]

        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.templates = json.load(f)
                    print(f"Loaded templates from {path}")
                    return
                except Exception as e:
                    print(f"Error loading templates from {path}: {e}")

        print("Warning: No editor_templates.json found.")
        self.templates = {}

    def get_template(self, key):
        return self.templates.get(key)

    def apply_template(self, key, card_context=None):
        """
        Returns a tuple (node_data, keywords_update, meta_info)
        node_data: The dict to populate the new tree item (EFFECT/COMMAND).
        keywords_update: Dict of keywords to set on the card.
        meta_info: Dict containing mapping info (e.g. condition_mapping).

        card_context: Optional dict containing 'civilizations', 'races' for substitution.
        """
        template = self.get_template(key)
        if not template:
            return None, {}, {}

        data = copy.deepcopy(template.get('data', {}))

        # Variable Substitution
        if card_context:
            self._recursive_substitute(data, card_context)

        # Ensure UIDs
        self._ensure_uids(data)

        keywords = template.get('keywords', {}).copy()
        meta = {
            "root_type": template.get('root_type', "EFFECT"),
            "condition_mapping": template.get('condition_mapping', {})
        }

        return data, keywords, meta

    def _recursive_substitute(self, obj, context):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Check for substitution pattern FIRST
                if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
                    val = v[0]
                    if val == "__CARD_CIVILIZATIONS__":
                        obj[k] = context.get('civilizations', ["FIRE"])
                        continue
                    elif val == "__CARD_RACES__":
                        obj[k] = context.get('races', ["Dragon"])
                        continue

                # Recurse if it's a container
                if isinstance(v, (dict, list)):
                    self._recursive_substitute(v, context)

        elif isinstance(obj, list):
            for item in obj:
                self._recursive_substitute(item, context)

    def _ensure_uids(self, obj):
        if isinstance(obj, dict):
            if 'uid' not in obj:
                obj['uid'] = str(uuid.uuid4())
            for v in obj.values():
                self._ensure_uids(v)
        elif isinstance(obj, list):
            for item in obj:
                self._ensure_uids(item)
