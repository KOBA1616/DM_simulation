# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.schema_def import CommandSchema, FieldSchema, FieldType, register_schema
from dm_toolkit.gui.i18n import tr

# Define constants for selection lists
MUTATION_TYPES = [
    "SPEED_ATTACKER", "BLOCKER", "SLAYER", "DOUBLE_BREAKER", "TRIPLE_BREAKER",
    "POWER_ATTACKER", "SHIELD_TRIGGER", "MACH_FIGHTER", "UNBLOCKABLE",
    "CANNOT_BE_BLOCKED", "ALWAYS_WIN_BATTLE", "INFINITE_POWER_ATTACKER",
    "JUST_DIVER", "G_STRIKE", "HYPER_ENERGY", "SHIELD_BURN", "EX_LIFE"
]

APPLY_MODIFIER_OPTIONS = MUTATION_TYPES + ["COST"]

MUTATION_KINDS_FOR_MUTATE = [
    "GIVE_POWER", "GIVE_ABILITY"
]

TARGET_SCOPES = [
    "PLAYER_SELF", "PLAYER_OPPONENT", "ALL"
]

DURATION_OPTIONS = [
    "THIS_TURN",
    "UNTIL_END_OF_OPPONENT_TURN",
    "UNTIL_START_OF_OPPONENT_TURN",
    "UNTIL_END_OF_YOUR_TURN",
    "UNTIL_START_OF_YOUR_TURN",
    "DURING_OPPONENT_TURN"
]

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
        f_links_out  # Enables output_value_key for discarded count
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
        FieldSchema("up_to", tr("Up To"), FieldType.BOOL, default=False),
        f_optional,
        f_links_in,
        f_links_out  # Enables output_value_key for card movement tracking
    ]))

    # MOVE_CARD
    register_schema(CommandSchema("MOVE_CARD", [
        f_target,
        f_filter,
        FieldSchema("to_zone", tr("Destination Zone"), FieldType.ZONE, default="GRAVEYARD"),
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        FieldSchema("up_to", tr("Up To"), FieldType.BOOL, default=False),
        f_optional,
        f_links_in,
        f_links_out
    ]))

    # REPLACE_CARD_MOVE
    register_schema(CommandSchema("REPLACE_CARD_MOVE", [
        f_target,
        f_filter,
        FieldSchema("from_zone", tr("Original Destination"), FieldType.ZONE, default="GRAVEYARD"),
        FieldSchema("to_zone", tr("Replacement Destination"), FieldType.ZONE, default="DECK_BOTTOM"),
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        FieldSchema("up_to", tr("Up To"), FieldType.BOOL, default=False),
        f_optional,
        f_links_in,
        f_links_out
    ]))

    # SEARCH_DECK
    register_schema(CommandSchema("SEARCH_DECK", [
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # LOOK_AND_ADD
    register_schema(CommandSchema("LOOK_AND_ADD", [
        f_filter,
        FieldSchema("amount", tr("Look Count"), FieldType.INT, default=3),
        FieldSchema("val2", tr("Add Count"), FieldType.INT, default=1),
        f_links_in,
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

    # PUT_CREATURE
    register_schema(CommandSchema("PUT_CREATURE", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in
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
        FieldSchema("mutation_kind", tr("Mutation Type"), FieldType.SELECT, options=MUTATION_KINDS_FOR_MUTATE),
        FieldSchema("amount", tr("Value (Power)"), FieldType.INT, default=0),
        FieldSchema("str_param", tr("Extra Param"), FieldType.STRING),
        # Using input_value_key for Duration Text (safe string storage)
        FieldSchema("input_value_key", tr("Duration"), FieldType.SELECT, options=DURATION_OPTIONS),
        f_links_in
    ]))

    # ADD_KEYWORD
    register_schema(CommandSchema("ADD_KEYWORD", [
        f_target,
        f_filter,
        # Keyword stored in str_param for correct C++ macro usage
        FieldSchema("str_param", tr("Keyword"), FieldType.SELECT, options=MUTATION_TYPES),
        # Using input_value_key for Duration Text (safe string storage)
        FieldSchema("input_value_key", tr("Duration"), FieldType.SELECT, options=DURATION_OPTIONS),
        # Hidden amount (default 0) to satisfy INT requirement
        FieldSchema("amount", tr("Amount"), FieldType.INT, default=0, widget_hint="hidden"),
        f_links_in
    ]))

    # APPLY_MODIFIER (Added)
    register_schema(CommandSchema("APPLY_MODIFIER", [
        f_target,
        f_filter,
        FieldSchema("str_param", tr("Effect ID"), FieldType.SELECT, options=APPLY_MODIFIER_OPTIONS),
        FieldSchema("amount", tr("Value (Modifier)"), FieldType.INT, default=1),
        FieldSchema("input_value_key", tr("Duration"), FieldType.SELECT, options=DURATION_OPTIONS),
        f_links_in
    ]))

    # PLAY_FROM_ZONE
    register_schema(CommandSchema("PLAY_FROM_ZONE", [
        FieldSchema("from_zone", tr("Source Zone"), FieldType.ZONE, default="HAND"),
        FieldSchema("to_zone", tr("Destination Zone"), FieldType.ZONE, default="BATTLE_ZONE"),
        FieldSchema("amount", tr("Max Cost"), FieldType.INT, default=99),
        FieldSchema("str_param", tr("Hint"), FieldType.STRING),
        FieldSchema("play_flags", tr("Play for Free"), FieldType.BOOL, default=False), # Mapped to checkbox
        f_links_in,
        f_links_out
    ]))

    # CAST_SPELL
    register_schema(CommandSchema("CAST_SPELL", [
        f_target,
        FieldSchema("target_filter", tr("Spell Filter"), FieldType.FILTER),
        f_links_in,
        f_links_out
    ]))

    # FRIEND_BURST
    register_schema(CommandSchema("FRIEND_BURST", [
        FieldSchema("str_param", tr("Race (e.g. Fire Bird)"), FieldType.STRING),
        FieldSchema("target_filter", tr("Friend Burst Condition"), FieldType.FILTER),
        f_links_in,
        f_links_out
    ]))

    # REVOLUTION_CHANGE
    register_schema(CommandSchema("REVOLUTION_CHANGE", [
        FieldSchema("target_filter", tr("Revolution Change Condition"), FieldType.FILTER)
    ]))

    # POWER_MOD
    register_schema(CommandSchema("POWER_MOD", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Power Adjustment"), FieldType.INT, default=0),
        f_links_in
    ]))

    # REVEAL_CARDS
    register_schema(CommandSchema("REVEAL_CARDS", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in
    ]))

    # SUMMON_TOKEN
    register_schema(CommandSchema("SUMMON_TOKEN", [
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        FieldSchema("str_param", tr("Token ID"), FieldType.STRING),
        f_links_in
    ]))

    # REGISTER_DELAYED_EFFECT
    register_schema(CommandSchema("REGISTER_DELAYED_EFFECT", [
        FieldSchema("str_param", tr("Effect ID"), FieldType.STRING),
        FieldSchema("amount", tr("Duration (Turns)"), FieldType.INT, default=1),
        f_links_in
    ]))

    # COST_REFERENCE
    register_schema(CommandSchema("COST_REFERENCE", [
        FieldSchema("ref_mode", tr("Reference Mode"), FieldType.SELECT, widget_hint="ref_mode_combo"),
        f_links_out
    ]))

    # RESOLVE_BATTLE
    register_schema(CommandSchema("RESOLVE_BATTLE", [
        f_target,
        f_links_in
    ]))

    # ADD_SHIELD
    register_schema(CommandSchema("ADD_SHIELD", [
        f_target,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in
    ]))

    # SEND_SHIELD_TO_GRAVE
    register_schema(CommandSchema("SEND_SHIELD_TO_GRAVE", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # SEARCH_DECK_BOTTOM
    register_schema(CommandSchema("SEARCH_DECK_BOTTOM", [
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # SEND_TO_DECK_BOTTOM
    register_schema(CommandSchema("SEND_TO_DECK_BOTTOM", [
        f_target,
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in
    ]))

    # LOOK_TO_BUFFER
    register_schema(CommandSchema("LOOK_TO_BUFFER", [
        f_filter,
        FieldSchema("amount", tr("Look Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # SELECT_FROM_BUFFER
    register_schema(CommandSchema("SELECT_FROM_BUFFER", [
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # PLAY_FROM_BUFFER
    register_schema(CommandSchema("PLAY_FROM_BUFFER", [
        f_filter,
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in,
        f_links_out
    ]))

    # MOVE_BUFFER_TO_ZONE
    register_schema(CommandSchema("MOVE_BUFFER_TO_ZONE", [
        FieldSchema("to_zone", tr("Destination Zone"), FieldType.ZONE, default="HAND"),
        FieldSchema("amount", tr("Count"), FieldType.INT, default=1),
        f_links_in
    ]))

    # GAME_RESULT
    register_schema(CommandSchema("GAME_RESULT", [
        FieldSchema("str_param", tr("Result (WIN/LOSE/DRAW)"), FieldType.STRING),
        f_links_in
    ]))

    # --- Logic Commands ---

    # CHOICE
    register_schema(CommandSchema("CHOICE", [
        f_amount,  # e.g. "Choose 1"
    ]))

    # SELECT_OPTION
    register_schema(CommandSchema("SELECT_OPTION", [
        FieldSchema("amount", tr("Selections Count"), FieldType.INT, default=1, min_value=1),
        FieldSchema("option_count", tr("Options Count"), FieldType.OPTIONS_CONTROL, default=1, min_value=1),
        FieldSchema("optional", tr("Allow Duplicates"), FieldType.BOOL, default=False)
    ]))

    # IF / IF_ELSE (Use filter as condition)
    # Outputs: condition result (0=false, 1=true) to output_value_key
    register_schema(CommandSchema("IF", [
        FieldSchema("target_filter", tr("Condition Filter"), FieldType.CONDITION_TREE),
        f_links_in,  # For dynamic condition inputs if needed
        f_links_out  # Enables output_value_key for condition result
    ]))
    register_schema(CommandSchema("IF_ELSE", [
        FieldSchema("target_filter", tr("Condition Filter"), FieldType.CONDITION_TREE),
        f_links_in,
        f_links_out  # Enables output_value_key for condition result
    ]))
    register_schema(CommandSchema("ELSE", [
        f_links_out
    ]))

    # SELECT_NUMBER
    register_schema(CommandSchema("SELECT_NUMBER", [
        FieldSchema("min_value", tr("Min Number"), FieldType.INT, default=1),
        FieldSchema("amount", tr("Max Number"), FieldType.INT, default=10),
        f_links_out
    ]))

    # FLOW
    register_schema(CommandSchema("FLOW", [
        FieldSchema("str_param", tr("Flow Instruction"), FieldType.STRING)
    ]))

    # NONE / Default
    register_schema(CommandSchema("NONE", []))
