from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter
from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils
from dm_toolkit.gui.editor.formatters.metadata_flags import SemanticMetadataFlags
from dm_toolkit.gui.i18n import tr

@register_formatter("TRANSITION")
class TransitionFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        to_z = command.get("to_zone", "")
        if to_z == "HAND":
            ctx.metadata[SemanticMetadataFlags.RETURNS_TO_HAND.value] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:

        from_z_raw = command.get("from_zone", "")
        if isinstance(from_z_raw, list):
            from_z_list = [CardTextResources.normalize_zone_name(z) for z in from_z_raw]
            from_z = from_z_list[0] if from_z_list else ""
            from_z_str = ZoneFormatter.format_zone_list(from_z_list, context="", joiner="と")
        else:
            from_z = CardTextResources.normalize_zone_name(from_z_raw)
            from_z_str = CardTextResources.get_zone_text(from_z) if from_z else ""

        to_z_raw = command.get("to_zone", "")
        if isinstance(to_z_raw, list):
            to_z_list = [CardTextResources.normalize_zone_name(z) for z in to_z_raw]
            to_z = to_z_list[0] if to_z_list else ""
            to_z_str = ZoneFormatter.format_zone_list(to_z_list, context="", joiner="、または")
        else:
            to_z = CardTextResources.normalize_zone_name(to_z_raw)
            to_z_str = CardTextResources.get_zone_text(to_z) if to_z else ""


        amount = get_command_amount(command, default=0)
        up_to_flag = bool(command.get('up_to', False))

        target_str, unit = cls._resolve_target(command, ctx)

        # For mapping lookup, we use the first element if it's a list. The actual text replacement uses from_z_str/to_z_str.
        template_key = (from_z, to_z)
        template = CardTextResources.get_zone_move_template(from_z, to_z)

        # Adjust template for up_to and all
        input_key = command.get('input_value_key') or command.get('input_link')
        is_all = (amount == 0 and not input_key)

        linked_text = None
        if input_key:
             linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list if ctx else [])

        formatted_qty = QuantityFormatter.format_quantity(amount, unit, up_to_flag, is_all, linked_text)

        # Handle face_up / face_down and enter_tapped modifiers uniformly
        face_modifier = ""
        if command.get("face_up"):
            face_modifier = "表向きにして"
        elif command.get("face_down"):
            face_modifier = "裏向きにして"

        tapped_modifier = ""
        if command.get("enter_tapped"):
            tapped_modifier = "タップして"

        combined_modifier = face_modifier + tapped_modifier

        if template_key == ("DECK", "HAND") and target_str == "カード":
             # Exception for generic deck drawing/searching phrasing
             if up_to_flag:
                  qty_str = TextUtils.format_up_to("{amount}", "枚", up_to=True)
                  template = f"山札からカードを{qty_str}選び、手札に{combined_modifier}加える。"
             else:
                  template = "{from_z}から{target}を{amount}{unit}選び、{to_z}に{combined_modifier}加える。"
        elif input_key and to_z == "DECK_BOTTOM" and CardTextResources.normalize_zone_name(from_z) == "HAND":
             # Handle input_key dynamic text for deck bottom generic returns from hand
             to_zone_text = CardTextResources.get_zone_text(to_z)
             scope = command.get("target_group") or command.get("scope", "NONE")
             owner = ""
             if scope in ["PLAYER_SELF", "SELF"]:
                 owner = "自分の"
             elif scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                 owner = "相手の"
             elif scope == "ALL_PLAYERS":
                 owner = "各プレイヤーの"

             template = f"{owner}手札から{{target}}を{formatted_qty}選び、{to_zone_text}に{combined_modifier}置く。"
        elif input_key and to_z == "DECK_BOTTOM":
             template = f"{{from_z}}の{{target}}を{formatted_qty}選び、{{to_z}}に{combined_modifier}置く。"
        else:
             from dm_toolkit.gui.editor.schema_def import get_schema
             schema = get_schema(command.get('type', ''))
             t_mode = schema.targeting_mode if schema and schema.targeting_mode else "NON_TARGET"
             template = QuantityFormatter.apply_to_template(template, formatted_qty, is_all, up_to_flag, to_z, from_z, modifier=combined_modifier, targeting_mode=t_mode)
        if "{from_z}" in template:
            template = template.replace("{from_z}", from_z_str)
        if "{to_z}" in template:
            template = template.replace("{to_z}", to_z_str)

        template = template.replace("{amount}", str(amount))

        text = template
        val2 = command.get("value2", 0)
        str_val = command.get("str_param") or command.get("str_val", "")
        dest_zone = command.get("destination_zone", "")
        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        src_zone = command.get("source_zone", "")
        src_str = CardTextResources.get_zone_text(src_zone) if src_zone else ""

        text = text.replace("{value1}", str(amount))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        if "{filter}" in text:
            text = text.replace("{filter}", target_str)

        if "{result}" in text:
            from dm_toolkit.gui.i18n import tr
            res = command.get("result", "")
            text = text.replace("{result}", tr(res))

        # Modifiers are now handled inside QuantityFormatter.apply_to_template before verbs are formed
        return text

@register_formatter("MOVE_CARD")
class MoveCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        # Convert MOVE_CARD command attributes to TRANSITION style logic for unified formatting
        dest_zone = command.get("destination_zone") or command.get("to_zone", "")
        src_zone = command.get("source_zone", "")

        # Build a temporary command dict resembling a TRANSITION command to reuse TransitionFormatter
        transition_cmd = command.copy()
        transition_cmd["from_zone"] = src_zone
        transition_cmd["to_zone"] = dest_zone

        # Delegate directly to TransitionFormatter to guarantee identical logic
        return TransitionFormatter.format(transition_cmd, ctx)

@register_formatter("REVEAL_TO_BUFFER")
class RevealToBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        src_zone = tr(command.get("from_zone", "DECK"))
        val1 = get_command_amount(command, default=0)
        amt = val1 if val1 > 0 else 1
        text = f"{src_zone}から{amt}枚を表向きにしてバッファに置く。"
        return text

@register_formatter("MOVE_BUFFER_TO_ZONE")
class MoveBufferToZoneFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        to_zone = tr(command.get("to_zone", "HAND"))
        filter_def = command.get("filter") or command.get("target_filter") or {}
        civs = filter_def.get("civilizations", []) if filter_def else []
        types = filter_def.get("types", []) if filter_def else []
        races = filter_def.get("races", []) if filter_def else []


        has_filter = bool(civs or types or races)
        val1 = get_command_amount(command, default=0)
        up_to = bool(command.get('up_to', False))
        is_all = (val1 == 0)

        if has_filter:
            civ_part = ""
            if civs:
                civ_part = "/".join(CardTextResources.get_civilization_text(c) for c in civs) + "の"

            if races:
                type_part = "/".join(races)
            elif "ELEMENT" in types:
                type_part = "エレメント"
            elif "SPELL" in types and "CREATURE" not in types:
                type_part = "呪文"
            elif "CREATURE" in types:
                type_part = "クリーチャー"
            else:
                type_part = "カード"

            qty_part = QuantityFormatter.format_quantity(val1, "枚", up_to, is_all)

            text = f"その中から、{civ_part}{type_part}を{qty_part}選び、{to_zone}に加える。"
            return text
        else:
            qty_part = QuantityFormatter.format_quantity(val1, "枚", up_to, is_all)
            text = f"その中から、{qty_part}を{to_zone}に加える。"
            return text

@register_formatter("REPLACE_CARD_MOVE")
class ReplaceCardMoveFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        dest_zone = command.get("destination_zone") or command.get("to_zone", "DECK_BOTTOM")
        src_zone = command.get("source_zone") or command.get("from_zone", "GRAVEYARD")

        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        orig_zone_str = CardTextResources.get_zone_text(src_zone) if src_zone else "元のゾーン"
        up_to_flag = bool(command.get('up_to', False))

        scope = command.get("target_group") or command.get("scope", "NONE")
        is_self_ref = scope == "SELF"

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list if ctx else [])

        from dm_toolkit.gui.editor.formatters.trigger_formatter import ReplacementEffectFormatter

        # Base replacement logic split into from_text and to_text
        to_text = f"{zone_str}に置く。"

        if linked_text:
            input_usage = str(command.get("input_value_usage") or command.get("input_usage") or "").upper()
            linked_target = "そのカード"
            if input_usage == "REPLACEMENT":
                 from_text = f"{orig_zone_str}に置く"
                 return ReplacementEffectFormatter.format_string(from_text, to_text)

            from_text = f"{linked_target}を{orig_zone_str}に置く"
            replaced = ReplacementEffectFormatter.format_string(from_text, to_text)
            return f"その後、{replaced}"

        amount = get_command_amount(command, default=0)
        target_str, unit = cls._resolve_target(command, ctx)

        if is_self_ref:
             from_text = f"このカードを{orig_zone_str}に置く"
             return ReplacementEffectFormatter.format_string(from_text, to_text)
        else:
             if amount > 0:
                  qty = TextUtils.format_up_to(amount, unit, up_to_flag)
                  prefix = f"{target_str}を{qty}選び、"
                  from_text = f"{orig_zone_str}に置く"
                  return prefix + ReplacementEffectFormatter.format_string(from_text, to_text)
             else:
                  from_text = f"{target_str}を{orig_zone_str}に置く"
                  return ReplacementEffectFormatter.format_string(from_text, to_text)
