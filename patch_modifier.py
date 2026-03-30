import re

with open("dm_toolkit/gui/editor/formatters/modifier_formatters.py", "r") as f:
    content = f.read()

# Replace all def format signatures
content = content.replace("def format(cls, cond: str, target: str, scope_prefix: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:", "def format(cls, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:")
content = content.replace("def format_characteristic(cls, behavior: CharacteristicModifierType, cond: str, target: str, scope_prefix: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:", "def format_characteristic(cls, behavior: CharacteristicModifierType, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:")
content = content.replace("def format(cls, mtype: str, cond: str, target: str, scope_prefix: str, value: int, modifier: Dict[str, Any], ctx: Any = None) -> str:", "def format(cls, mtype: str, cond: str, target: str, value: Any, modifier: Dict[str, Any], ctx: Any = None) -> str:")

# Fix the method calls
content = content.replace("return cls.format_characteristic(CharacteristicModifierType.GRANT, cond, target, scope_prefix, value, modifier, ctx)", "return cls.format_characteristic(CharacteristicModifierType.GRANT, cond, target, value, modifier, ctx)")
content = content.replace("return cls.format_characteristic(CharacteristicModifierType.SET, cond, target, scope_prefix, value, modifier, ctx)", "return cls.format_characteristic(CharacteristicModifierType.SET, cond, target, value, modifier, ctx)")
content = content.replace("return cls.format_characteristic(CharacteristicModifierType.RESTRICT, cond, target, scope_prefix, value, modifier, ctx)", "return cls.format_characteristic(CharacteristicModifierType.RESTRICT, cond, target, value, modifier, ctx)")

content = content.replace("return formatter.format(cond, target, scope_prefix, value, modifier, ctx)", "return formatter.format(cond, target, value, modifier, ctx)")
content = content.replace("return f\"{cond}{scope_prefix}常在効果: {tr(mtype)}\"", "return f\"{cond}{target}常在効果: {tr(mtype)}\"")
content = content.replace("{scope_prefix}{keyword}を与える", "{target}に{keyword}を与える")
content = content.replace("{scope_prefix}制限を与える", "{target}に制限を与える")

with open("dm_toolkit/gui/editor/formatters/modifier_formatters.py", "w") as f:
    f.write(content)
