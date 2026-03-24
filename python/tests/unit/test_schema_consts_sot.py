import inspect
from dm_toolkit import consts
from dm_toolkit.gui.editor import schema_config
from dm_toolkit.gui.editor import schema_def


def test_schema_uses_consts_for_delayed_effects():
    # Ensure consts defines DELAYED_EFFECT_IDS
    assert hasattr(consts, 'DELAYED_EFFECT_IDS')
    assert isinstance(consts.DELAYED_EFFECT_IDS, list) and len(consts.DELAYED_EFFECT_IDS) > 0

    # Ensure schema_config exposes DELAYED_EFFECT_IDS and it matches consts (SSOT)
    assert hasattr(schema_config, 'DELAYED_EFFECT_IDS')
    assert list(schema_config.DELAYED_EFFECT_IDS) == list(consts.DELAYED_EFFECT_IDS)

    # Register schemas and verify REGISTER_DELAYED_EFFECT uses consts options
    schema_def.SCHEMA_REGISTRY.clear()
    schema_config.register_all_schemas()
    schema = schema_def.get_schema('REGISTER_DELAYED_EFFECT')
    assert schema is not None
    # Find field 'str_param' in the schema fields
    str_param_fields = [f for f in schema.fields if f.key == 'str_param']
    assert str_param_fields, 'REGISTER_DELAYED_EFFECT must have str_param field'
    f = str_param_fields[0]
    # The options should come from consts.DELAYED_EFFECT_IDS
    assert list(f.options) == list(consts.DELAYED_EFFECT_IDS)
