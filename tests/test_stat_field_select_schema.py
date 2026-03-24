# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.schema_config import register_all_schemas
from dm_toolkit.gui.editor.schema_def import get_schema, FieldType
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_stat_field_is_select_and_has_options() -> None:
    """STAT コマンドの str_param は SELECT になっており、選択肢に STAT_KEY_MAP のキーが含まれることを検証する。"""
    # Ensure registry is populated
    register_all_schemas()
    schema = get_schema("STAT")
    assert schema is not None, "STAT schema が登録されていません"

    # Find the str_param field
    fields = {f.key: f for f in schema.fields}
    assert "str_param" in fields, "STAT schema に str_param フィールドがありません"
    f = fields["str_param"]

    # Expect SELECT field type and options containing at least one known stat key
    assert f.field_type == FieldType.SELECT, "STAT.str_param は SELECT フィールドである必要があります"
    known_keys = set(CardTextResources.STAT_KEY_MAP.keys())
    assert any(opt in known_keys for opt in f.options), "STAT.str_param の options に STAT_KEY_MAP のキーが含まれていません"
