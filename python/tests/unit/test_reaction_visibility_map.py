from dm_toolkit.gui.editor.forms.parts.reaction_condition_widget import REACTION_VISIBILITY_MAP if 'REACTION_VISIBILITY_MAP' in globals() else None


def test_reaction_visibility_map_present():
    # The module should define REACTION_VISIBILITY_MAP or similar mapping via code
    # Since the mapping is local to the function, we test behavior by importing the widget and calling update_visibility
    from dm_toolkit.gui.editor.forms.parts.reaction_condition_widget import ReactionConditionWidget
    w = ReactionConditionWidget()
    # Ensure no exception when updating visibility for known types
    w.update_visibility('STRIKE_BACK')
    w.update_visibility('NINJA_STRIKE')
    w.update_visibility('REVOLUTION_0_TRIGGER')
    # Basic assertions on attribute existence
    assert hasattr(w, 'label_mana')
    assert hasattr(w, 'mana_min_spin')
    assert hasattr(w, 'shield_civ_match_check')
