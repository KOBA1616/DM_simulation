# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.schema_def import CommandSchema, FieldSchema, FieldType
from dm_toolkit.gui.localization import tr

CARD_SCHEMA = CommandSchema(
    type_name="CARD",
    fields=[
        FieldSchema(key="id", label=tr("ID"), field_type=FieldType.INT, min_value=0, max_value=9999),
        FieldSchema(key="name", label=tr("Name"), field_type=FieldType.STRING, tooltip=tr("Enter card name...")),
        FieldSchema(key="civilizations", label=tr("Civilization"), field_type=FieldType.CIVILIZATION),
        FieldSchema(key="type", label=tr("Type"), field_type=FieldType.TYPE_SELECT, tooltip=tr("Card type (Creature, Spell, etc.)")),
        FieldSchema(key="cost", label=tr("Cost"), field_type=FieldType.INT, min_value=0, max_value=99, tooltip=tr("Mana cost of the card")),
        FieldSchema(key="power", label=tr("Power"), field_type=FieldType.INT, min_value=0, max_value=99999, tooltip=tr("Creature power (ignored for Spells)"), visible_if={"type": ["CREATURE", "EVOLUTION_CREATURE", "NEO_CREATURE", "G_NEO_CREATURE"]}),
        FieldSchema(key="races", label=tr("Races"), field_type=FieldType.RACES, tooltip=tr("Comma-separated list of races (e.g. 'Dragon, Fire Bird')")),
        FieldSchema(key="evolution_condition", label=tr("Evolution Condition"), field_type=FieldType.STRING, tooltip=tr("e.g. Fire Bird"), visible_if={"type": ["EVOLUTION_CREATURE", "NEO_CREATURE", "G_NEO_CREATURE"]}),
        FieldSchema(key="is_key_card", label=tr("Is Key Card"), field_type=FieldType.BOOL, tooltip=tr("Mark this card as critical for the deck's strategy.")),
        FieldSchema(key="ai_importance_score", label=tr("AI Importance Score"), field_type=FieldType.INT, min_value=0, max_value=1000, tooltip=tr("Higher values (0-1000) prioritize this card for AI protection and targeting.")),
        FieldSchema(key="hyper_energy", label=tr("Hyper Energy"), field_type=FieldType.BOOL, tooltip=tr("Enables Hyper Energy cost reduction logic.")),
        FieldSchema(key="twinpact", label=tr("Twinpact"), field_type=FieldType.BOOL, tooltip=tr("Enable to generate a Spell Side node in the logic tree.")),
    ]
)
