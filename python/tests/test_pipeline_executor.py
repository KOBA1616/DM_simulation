import pytest
import dm_ai_module

def test_pipeline_search_deck():
    f = dm_ai_module.FilterDef()
    f.zones = ["DECK"]

    a = dm_ai_module.ActionDef(dm_ai_module.EffectActionType.SEARCH_DECK, dm_ai_module.TargetScope.NONE, f)
    a.value1 = 1

    # Ensure explicit NONE condition to avoid auto-IF
    c = dm_ai_module.ConditionDef()
    c.type = "NONE"

    e = dm_ai_module.EffectDef(dm_ai_module.TriggerType.NONE, c, [a])

    instructions = dm_ai_module.LegacyJsonAdapter.convert(e)

    # Logic: 1. SELECT, 2. MOVE (to hand), 3. MODIFY (Shuffle)
    assert len(instructions) == 3
