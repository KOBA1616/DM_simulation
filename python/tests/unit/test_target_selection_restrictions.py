import pytest
import dm_ai_module


def _require_native_or_skip():
    # python/tests/conftest.py may replace dm_ai_module.GameState with a Python wrapper.
    # The wrapper breaks the type-based detection in the existing require_native fixture.
    # Detect native availability by checking whether we can obtain a native state object.
    try:
        gs = dm_ai_module.GameState(1)
    except Exception:
        pytest.skip("dm_ai_module not available")

    native = getattr(gs, "_native", gs)
    if not hasattr(native, "players"):
        pytest.skip("Test requires native C++ dm_ai_module extension")


def _setup_state_with_two_opponent_creatures():
    _require_native_or_skip()
    # Build a minimal CardDatabase locally.
    # Use card_id=0 so SelectionSystem accepts them as generic dummy cards even without CardRegistry entries.
    card_db = dm_ai_module.CardDatabase()
    cdef = dm_ai_module.CardDefinition()
    cdef.id = 0
    cdef.name = "Dummy"
    cdef.type = dm_ai_module.CardType.CREATURE
    cdef.cost = 1
    cdef.power = 1000
    card_db[0] = cdef

    gs = dm_ai_module.GameState(40)
    native = getattr(gs, "_native", gs)
    native.active_player_id = 0

    # Ensure zones are clean
    native.players[0].battle_zone.clear()
    native.players[1].battle_zone.clear()

    # Source on player 0
    src_inst = dm_ai_module.CardInstance()
    src_inst.instance_id = 500
    src_inst.card_id = 0
    src_inst.owner = 0
    native.players[0].battle_zone.append(src_inst)
    native.register_card_instance(src_inst)

    # Two opponent candidates
    opp_a = dm_ai_module.CardInstance()
    opp_a.instance_id = 600
    opp_a.card_id = 0
    opp_a.owner = 1

    opp_b = dm_ai_module.CardInstance()
    opp_b.instance_id = 601
    opp_b.card_id = 0
    opp_b.owner = 1

    native.players[1].battle_zone.append(opp_a)
    native.players[1].battle_zone.append(opp_b)
    native.register_card_instance(opp_a)
    native.register_card_instance(opp_b)

    return native, card_db, src_inst.instance_id, opp_a.instance_id, opp_b.instance_id


def _push_target_select_pending_effect(native_state, card_db, source_instance_id):
    # Build a PendingEffect (ResolveType::TARGET_SELECT) so PendingEffectStrategy is used.
    f = dm_ai_module.FilterDef()
    f.zones = ["BATTLE_ZONE"]
    f.types = ["CREATURE"]
    f.owner = "OPPONENT"
    f.count = 1

    native_state.push_pending_target_select(source_instance_id, 0, f, 1)

    assert native_state.get_pending_effect_count() > 0


def test_cannot_be_selected_is_excluded_from_pending_strategy_actions():
    native, card_db, source_id, restricted_id, allowed_id = _setup_state_with_two_opponent_creatures()

    pe = dm_ai_module.PassiveEffect()
    pe.type = dm_ai_module.PassiveType.CANNOT_BE_SELECTED
    pe.controller = 0
    pe.specific_targets = [restricted_id]
    native.add_passive_effect(pe)

    _push_target_select_pending_effect(native, card_db, source_id)

    actions = dm_ai_module.IntentGenerator.generate_legal_actions(native, card_db)
    selected_ids = [a.target_instance_id for a in actions if a.type == dm_ai_module.ActionType.SELECT_TARGET]

    assert restricted_id not in selected_ids
    assert allowed_id in selected_ids


def test_force_selection_filters_to_forced_targets_for_opponent_selection():
    native, card_db, source_id, other_id, forced_id = _setup_state_with_two_opponent_creatures()

    pe = dm_ai_module.PassiveEffect()
    pe.type = dm_ai_module.PassiveType.FORCE_SELECTION
    pe.controller = 0
    pe.specific_targets = [forced_id]
    native.add_passive_effect(pe)

    _push_target_select_pending_effect(native, card_db, source_id)

    actions = dm_ai_module.IntentGenerator.generate_legal_actions(native, card_db)
    selected_ids = [a.target_instance_id for a in actions if a.type == dm_ai_module.ActionType.SELECT_TARGET]

    assert forced_id in selected_ids
    assert other_id not in selected_ids
