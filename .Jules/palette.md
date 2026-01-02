## 2024-05-22 - Action Editor Tooltips
**Learning:** In complex configuration forms (like a card editor), dropdown options can be ambiguous. Adding explicit tooltips and helper text significantly reduces the cognitive load for users who might not know the internal jargon (like "MEKRAID" or "COST_REFERENCE").
**Action:** When designing configuration UIs, always include a mechanism for descriptive helper text alongside technical selection fields. Using a dedicated `QLabel` below the input provides better accessibility than a hover-only tooltip.
