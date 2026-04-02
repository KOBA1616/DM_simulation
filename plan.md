1. **Issue 14: Composite type/race handling in `FilterTextFormatter`**
   - Create a new `CompositeTypeGenerator` (or enhance `FilterTextFormatter` / `TargetResolutionService`) to generate proper compound types like "ドラゴン・エレメント", "タマシード/クリーチャー" instead of joining with "の".
   - Update `FilterTextFormatter.describe_simple_filter` to use this logic for generating composite type text correctly.

2. **Issue 15: `ContextMerger` pattern matching via JSON/Dict**
   - Refactor `ContextMerger` in `context_merger.py` to use declarative, data-driven pattern definitions (e.g., matching a sequence of command types `['CAST_SPELL', 'REPLACE_CARD_MOVE']`) instead of Python function conditions (`cond_spell_then_replace`).

3. **Issue 16: Clarify target of `RESET_INSTANCE`**
   - Update `ResetInstanceFormatter` in `game_action_formatters.py`.
   - Read the target/reset mode properties (e.g., if it's unsealing, re-constructing, or ignoring modifiers) and generate corresponding rule text ("封印を外す", "再構築する", "無視する") instead of the hardcoded "状態を初期化する（効果を無視する）".
   - Add the necessary property fields in `schema_config.py` for `RESET_INSTANCE`.

4. **Issue 17: Structural AST separation in `ReplacementEffectFormatter`**
   - Refactor `ReplacementEffectFormatter` in `trigger_formatter.py`.
   - Instead of formatting sub-actions and string concatenating "かわりに", structure it to format a `trigger_condition` node and a `replacement_actions` node properly, handling their AST nodes separately.
   - Update schema definition for `REPLACEMENT_EFFECT` to use `replaced_action` and `replacement_action` explicitly if needed, or fix how the formatter parses them.

5. **Issue 18: Clean dispatch for `REVOLUTION_CHANGE`**
   - Remove the hardcoded fallback for `REVOLUTION_CHANGE` inside `CommandFormatterRegistry.format_command` (`command_registry.py`).
   - Create/Update a formatter for `REVOLUTION_CHANGE` or dispatch to `INVOKE_KEYWORD` (or have `SpecialKeywordFormatterBase` format it naturally).

6. **Issue 19: Nested conditions in `ClauseJoiner`**
   - Refactor `ClauseJoiner.join_condition_ast` in `clause_joiner.py` to count clause depth. If nested deeply, use variations like "〜で、", "〜であり、かつ" to produce natural Japanese based on sentence length/depth.

7. **Issue 20: Separate UI labels and rule text in `InputLinkFormatter`**
   - Refactor `InputLinkFormatter.format_input_source_label` to only fetch the label.
   - Move the logic that strips out UI markers (like removing content after `"("` or `"（"`) to a separate normalizer (e.g., a `Normalizer` class or before `CardTextGenerator` applies it) so UI labels are kept distinct from rule text data.

8. **Pre-commit checks**
   - Run `pre_commit_instructions` tool to verify steps before submit.
