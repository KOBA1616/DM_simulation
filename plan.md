1. **Extract `VariableLinkTextFormatter` for input usage labels**
   - Create a new file `dm_toolkit/gui/editor/formatters/variable_link_formatter.py` (or update `utils.py`) to hold a new `VariableLinkTextFormatter` class.
   - Move `_format_input_usage_label`, `_format_input_source_label`, `_normalize_linked_count_label`, `_format_linked_count_token`, `_infer_output_value_label`, and `_format_input_link_context_suffix` from `CardTextGenerator` to `VariableLinkTextFormatter`.
   - Update `CardTextGenerator` to delegate to `VariableLinkTextFormatter` for all input usage label generation to separate concerns.
2. **Implement Special Keyword Handlers (Plugin System)**
   - Create `dm_toolkit/gui/editor/formatters/keyword_registry.py` with a `SpecialKeywordRegistry` class to manage special keywords.
   - Define a `SpecialKeywordFormatterBase` interface.
   - Implement handlers for `revolution_change`, `friend_burst`, and `mekraid` (currently hardcoded in `generate_body_text`).
   - Register these handlers dynamically.
   - Update `CardTextGenerator.generate_body_text` to iterate over special keywords using the registry, removing the hardcoded `if k == "revolution_change"` blocks.
3. **Verify Refactoring**
   - Run tests (especially UI/text generation tests if they exist) and ensure no functionality is broken.
   - Call `pre_commit_instructions` to ensure proper testing, verification, review, and reflection are done.
4. **Submit**
   - Commit and submit the changes.
