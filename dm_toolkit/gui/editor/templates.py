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

    def apply_template(self, key, card_context=None, extra_context=None):
        """
        Returns a tuple (node_data, keywords_update, meta_info)
        node_data: The dict to populate the new tree item (EFFECT/COMMAND).
        keywords_update: Dict of keywords to set on the card.
        meta_info: Dict containing mapping info (e.g. condition_mapping).

        card_context: Optional dict containing 'civilizations', 'races' for substitution.
        extra_context: Optional dict for additional substitution keys (e.g. 'fb_races', 'rc_races').
        """
        template = self.get_template(key)
        if not template:
            return None, {}, {}

        data = copy.deepcopy(template.get('data', {}))

        # Merge extra_context into card_context for substitution
        merged_context = {}
        if card_context:
            merged_context.update(card_context)
        if extra_context:
            merged_context.update(extra_context)

        # Variable Substitution
        if merged_context:
            self._recursive_substitute(data, merged_context)

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
                    # 再発防止: フレンドバースト/革命チェンジ用種族プレースホルダーのリスト形式置換。
                    # テンプレートで ["__FB_RACES__"] などと書くと種族リストに展開。
                    elif val == "__FB_RACES__":
                        obj[k] = context.get('fb_races', context.get('races', []))
                        continue
                    elif val == "__RC_RACES__":
                        obj[k] = context.get('rc_races', context.get('races', []))
                        continue
                    elif val == "__MEKRAID_RACES__":
                        obj[k] = context.get('mekraid_races', context.get('races', []))
                        continue
                    elif val == "__DD_CIVS__":
                        obj[k] = context.get('dd_civs', [])
                        continue
                    # LOOK_SELECT_TO_ZONE テンプレート用フィルタリストプレースホルダー
                    # 再発防止: ["__TEMPLATE_CIVS__"] など1要素リストとして書くとリストに展開される。
                    elif val == "__TEMPLATE_CIVS__":
                        obj[k] = context.get('template_civs', [])
                        continue
                    elif val == "__TEMPLATE_RACES__":
                        obj[k] = context.get('template_races', [])
                        continue
                    elif val == "__TEMPLATE_TYPES__":
                        obj[k] = context.get('template_types', [])
                        continue

                # Check for string substitution (e.g. str_param: "__CARD_RACES__")
                # 再発防止: str_param に単値プレースホルダーが書かれた場合は先頭要素を文字列として返す。
                if isinstance(v, str):
                    if v == "__CARD_RACES__":
                        races = context.get('races', [])
                        obj[k] = races[0] if races else ""
                        continue
                    elif v == "__CARD_CIVILIZATIONS__":
                        civs = context.get('civilizations', [])
                        obj[k] = civs[0] if civs else ""
                        continue
                    elif v == "__FB_RACES__":
                        fb = context.get('fb_races', context.get('races', []))
                        obj[k] = fb[0] if fb else ""
                        continue
                    elif v == "__RC_RACES__":
                        rc = context.get('rc_races', context.get('races', []))
                        obj[k] = rc[0] if rc else ""
                        continue
                    elif v == "__MEKRAID_RACES__":
                        mk = context.get('mekraid_races', context.get('races', []))
                        obj[k] = mk[0] if mk else ""
                        continue
                    elif v == "__DD_CIVS__":
                        dd_civs = context.get('dd_civs', [])
                        obj[k] = dd_civs[0] if dd_civs else ""
                        continue
                    elif v == "__DD_COST__":
                        obj[k] = int(context.get('cost', 0))
                        continue
                    elif v == "__DD_TEXT__":
                        obj[k] = str(context.get('raw_text', ""))
                        continue
                    # 再発防止: LOOK_SELECT_TO_ZONE 用 int/str プレースホルダー。
                    # 文字列として書かれた値を int/str に型変換して返す。
                    elif v == "__LOOK_AMOUNT__":
                        obj[k] = int(context.get('look_amount', 4))
                        continue
                    elif v == "__SELECT_AMOUNT__":
                        obj[k] = int(context.get('select_amount', -1))
                        continue
                    elif v == "__TO_ZONE__":
                        obj[k] = str(context.get('to_zone', "HAND"))
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
