import pytest

from dm_toolkit import consts
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_mutationkind_handlers_cover_expected_members():
    """Ensure MutationKind additions are either handled by MUTATE_KIND_HANDLERS
    or intentionally classified as keyword/effect types listed in consts.
    This test will fail when a new MutationKind is added without updating
    the handler map or the approved whitelist.
    """
    # Gather handler keys (they are enum members)
    handler_keys = set(CardTextGenerator.MUTATE_KIND_HANDLERS.keys())
    handler_names = {k.name for k in handler_keys}

    # All defined enum members
    all_names = {m.name for m in consts.MutationKind}

    # Allowed unhandled sets: mutation types and grantable keywords (uppercased), effect ids
    allowed_unhandled = set()
    allowed_unhandled.update([n for n in consts.MUTATION_TYPES])
    # GRANTABLE_KEYWORDS may contain mixed-case; normalize to upper
    allowed_unhandled.update([s.upper() for s in consts.GRANTABLE_KEYWORDS])
    allowed_unhandled.update([e for e in consts.EFFECT_IDS])

    # Include known special cases that are safe to be unhandled by MUTATE_KIND_HANDLERS
    allowed_unhandled.update({
        "REVOLUTION_CHANGE",
        "TARGET_THIS_FORCE_SELECT",
        "TARGET_THIS_CANNOT_SELECT",
    })

    missing = all_names - handler_names - allowed_unhandled

    assert missing == set(), (
        "Found MutationKind members without handlers or whitelist entries: "
        + ", ".join(sorted(missing))
    )
