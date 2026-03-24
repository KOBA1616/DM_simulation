# -*- coding: utf-8 -*-
"""
Contract tests: ensure ConditionEditorWidget references the
single-source condition/type definitions in CardTextResources
and schema_config.
"""
from dm_toolkit.gui.editor.forms.parts import condition_widget as editor_parts
from dm_toolkit.gui.editor import text_resources
from dm_toolkit.gui.editor import schema_config


def test_condition_templates_use_canonical_condition_types():
    """All condition templates must reference a condition `type`
    that is declared in CardTextResources.CONDITION_TYPE_LABELS.
    """
    ct_labels = set(text_resources.CardTextResources.CONDITION_TYPE_LABELS.keys())
    for key, meta in editor_parts.CONDITION_TEMPLATES.items():
        ctype = meta.get("data", {}).get("type")
        assert ctype in ct_labels, f"Template {key} references unknown condition type: {ctype}"


def test_condition_ui_config_keys_present_in_canonical_sources():
    """Keys used by the legacy CONDITION_UI_CONFIG must appear in the
    canonical CardTextResources labels or in the declarative
    schema_config.CONDITION_FORM_SCHEMA.
    """
    canonical = set(text_resources.CardTextResources.CONDITION_TYPE_LABELS.keys())
    canonical |= set(schema_config.CONDITION_FORM_SCHEMA.keys())
    # Allow CUSTOM as a UI-configurable escape hatch
    canonical.add("CUSTOM")

    for k in editor_parts.CONDITION_UI_CONFIG.keys():
        assert k in canonical, f"CONDITION_UI_CONFIG contains unknown key: {k}"
