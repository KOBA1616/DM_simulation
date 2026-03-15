from dm_toolkit import consts
from dm_toolkit.gui.editor import logic_mask
from dm_toolkit.gui.editor import data_manager_helpers


def test_logic_mask_triggers_subset_of_consts():
    # Ensure the UI-level trigger lists are subsets of the canonical definition
    # Allow legacy synonyms (e.g. AT_END_OF_TURN -> ON_TURN_END)
    synonym_map = {
        'AT_END_OF_TURN': 'ON_TURN_END'
    }

    for t in logic_mask.ALL_TRIGGERS:
        if t in consts.TRIGGER_TYPES:
            continue
        mapped = synonym_map.get(t)
        assert mapped and mapped in consts.TRIGGER_TYPES, f"Trigger '{t}' is not normalized to canonical TRIGGER_TYPES"

    for t in logic_mask.SPELL_TRIGGERS:
        assert (t in consts.SPELL_TRIGGER_TYPES) or (t in consts.TRIGGER_TYPES)


def test_default_trigger_is_valid():
    # data_manager_helpers.create_default_trigger_data is a small helper that
    # returns a dict with a 'trigger' key set to a default value.
    d = data_manager_helpers.create_default_trigger_data(None)
    assert isinstance(d, dict)
    trig = d.get('trigger')
    assert trig in consts.TRIGGER_TYPES
