# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.schema_def import CommandSchema, FieldSchema, FieldType, register_schema
from dm_toolkit.gui.localization import tr

def register_all_schemas():
    """
    Registers all Command UI schemas.
    This replaces the imperative logic in ActionEditForm.
    """

    # Common fields
    f_target = FieldSchema("target_group", tr("Target"), FieldType.PLAYER, default="PLAYER_SELF")
    f_filter = FieldSchema("target_filter", tr("Filter"), FieldType.FILTER)
    f_amount = FieldSchema("amount", tr("Amount"), FieldType.INT, default=1, min_value=1)
    f_optional = FieldSchema("optional", tr("Optional"), FieldType.BOOL, default=False)
    f_links_out = FieldSchema("links", tr("Variable Links"), FieldType.LINK, produces_output=True)
    f_links_in = FieldSchema("links", tr("Variable Links"), FieldType.LINK, produces_output=False)

    # DRAW_CARD
    register_schema(CommandSchema("DRAW_CARD", [
        f_target,
        FieldSchema("amount", tr("Cards to Draw"), FieldType.INT, default=1, min_value=1),
        f_optional,
        FieldSchema("up_to", tr("Up To"), FieldType.BOOL, default=False),
        f_links_out
    ]))

    # DISCARD
    # Outputs: discarded count to output_value_key (card IDs in future C++ enhancement)
    register_schema(CommandSchema("DISCARD", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1, min_value=1),
        FieldSchema("up_to", tr("Up To"), FieldType.BOOL, default=False),
        f_optional,
        f_links_out  # Enables output_value_key generation for discarded count
    ]))

    # DESTROY / MANA_CHARGE / RETURN_TO_HAND / BREAK_SHIELD
    # Outputs: moved/destroyed/returned count and card instance IDs to output_value_key
    for cmd in ["DESTROY", "MANA_CHARGE", "RETURN_TO_HAND", "BREAK_SHIELD"]:
        register_schema(CommandSchema(cmd, [
            f_target,
            f_filter,
            FieldSchema("amount", tr("Count (if selecting)"), FieldType.INT, default=1),
            f_links_out  # Enables output_value_key for card movement tracking
        ]))

    # TAP / UNTAP (state changes, no movement)
    for cmd in ["TAP", "UNTAP"]:
        register_schema(CommandSchema(cmd, [
            f_target,
            f_filter,
            f_links_in  # Input only - state changes don't need output tracking
        ]))

    # SHIELD_TRIGGER / SHUFFLE_DECK (no card selection)
    for cmd in ["SHIELD_TRIGGER", "SHUFFLE_DECK"]:
        register_schema(CommandSchema(cmd, [
            f_target,
            f_filter,
            f_links_in
        ]))

    # TRANSITION (Move Card)
    # Outputs: moved count and card instance IDs to output_value_key
    register_schema(CommandSchema("TRANSITION", [
        f_target,
        f_filter,
        FieldSchema("from_zone", tr("Source Zone"), FieldType.ZONE, default="NONE"),
        FieldSchema("to_zone", tr("Destination Zone"), FieldType.ZONE, default="HAND"),
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_optional,
        f_links_out  # Enables output_value_key for card movement tracking
    ]))

    # SEARCH_DECK
    register_schema(CommandSchema("SEARCH_DECK", [
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_out
    ]))

    # LOOK_AND_ADD
    register_schema(CommandSchema("LOOK_AND_ADD", [
        f_filter,
        FieldSchema("amount", tr("Look Count"), FieldType.INT, default=3),
        FieldSchema("val2", tr("Add Count"), FieldType.INT, default=1),
        f_links_out
    ]))

    # MEKRAID
    register_schema(CommandSchema("MEKRAID", [
        f_filter,
        FieldSchema("amount", tr("Level (Max Cost)"), FieldType.INT, default=7),
        FieldSchema("val2", tr("Look Count"), FieldType.INT, default=3),
        FieldSchema("select_count", tr("Select Count"), FieldType.INT, default=1),
        f_links_out
    ]))

    # QUERY
    register_schema(CommandSchema("QUERY", [
        f_target,
        f_filter,
        FieldSchema("query_mode", tr("Query Mode"), FieldType.SELECT, options=[
            "COUNT_CARDS", "GET_GAME_STAT", "HAS_TARGET"
        ]),
        f_links_out
    ]))

    # MUTATE (Grant Power/Keyword)
    register_schema(CommandSchema("MUTATE", [
        f_target,
        f_filter,
        FieldSchema("mutation_kind", tr("Mutation Type"), FieldType.STRING), # Or special widget
        FieldSchema("amount", tr("Duration/Value"), FieldType.INT, default=0),
        FieldSchema("str_param", tr("Extra Param"), FieldType.STRING)
    ]))

    # ADD_KEYWORD
    register_schema(CommandSchema("ADD_KEYWORD", [
        FieldSchema("target_group", tr("Target Scope"), FieldType.SELECT, default="PLAYER_SELF",
                   options=["PLAYER_SELF", "PLAYER_OPPONENT", "ALL"]), # Simplified options
        f_filter,
        FieldSchema("mutation_kind", tr("Keyword"), FieldType.STRING),
        FieldSchema("amount", tr("Duration (Turns)"), FieldType.INT, default=1)
    ]))

    # PLAY_FROM_ZONE
    register_schema(CommandSchema("PLAY_FROM_ZONE", [
        FieldSchema("from_zone", tr("Source Zone"), FieldType.ZONE, default="HAND"),
        FieldSchema("to_zone", tr("Destination Zone"), FieldType.ZONE, default="BATTLE_ZONE"),
        FieldSchema("amount", tr("Max Cost"), FieldType.INT, default=99),
        FieldSchema("str_param", tr("Hint"), FieldType.STRING),
        FieldSchema("play_flags", tr("Play for Free"), FieldType.BOOL, default=False), # Mapped to checkbox
        f_links_out
    ]))

    # CAST_SPELL
    register_schema(CommandSchema("CAST_SPELL", [
        f_target,
        FieldSchema("target_filter", tr("Spell Filter"), FieldType.FILTER),
        f_links_out
    ]))

    # REVOLUTION_CHANGE
    register_schema(CommandSchema("REVOLUTION_CHANGE", [
        FieldSchema("target_filter", tr("Revolution Change Condition"), FieldType.FILTER)
    ]))

    # --- Logic Commands ---

    # CHOICE
    register_schema(CommandSchema("CHOICE", [
        f_amount,  # e.g. "Choose 1"
    ]))

    # SELECT_OPTION
    register_schema(CommandSchema("SELECT_OPTION", [
        f_amount,
        FieldSchema("str_param", tr("Option Text"), FieldType.STRING)
    ]))

    # IF / IF_ELSE (Use filter as condition)
    register_schema(CommandSchema("IF", [
        FieldSchema("target_filter", tr("Condition Filter"), FieldType.FILTER)
    ]))
    register_schema(CommandSchema("IF_ELSE", [
        FieldSchema("target_filter", tr("Condition Filter"), FieldType.FILTER)
    ]))
    register_schema(CommandSchema("ELSE", []))

    # SELECT_NUMBER
    register_schema(CommandSchema("SELECT_NUMBER", [
        FieldSchema("amount", tr("Max Number"), FieldType.INT, default=10),
        f_links_out
    ]))

    # FLOW
    register_schema(CommandSchema("FLOW", [
        FieldSchema("str_param", tr("Flow Instruction"), FieldType.STRING)
    ]))

    # NONE / Default
    register_schema(CommandSchema("NONE", []))
