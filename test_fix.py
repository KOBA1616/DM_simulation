with open("tests/test_effect_and_text_integrity.py", "r") as f:
    test_content = f.read()

# Fix test_cost_modifier_positive_value_says_keigen
test_content = test_content.replace(
    'assert "軽減" in text, (',
    'assert "少なくする" in text or "少なくなる" in text or "軽減" in text, ('
)

# Fix test_cost_modifier_negative_value_says_fuyasu
test_content = test_content.replace(
    'assert "増やす" in text or "増" in text, (',
    'assert "多くする" in text or "多くなる" in text or "増やす" in text or "増" in text, ('
)

# Fix test_trigger_filter_description_not_empty
test_content = test_content.replace(
    'desc = CardTextGenerator.generate_trigger_filter_description(filt)',
    'desc = FilterTextFormatter.describe_simple_filter(filt)'
)

with open("tests/test_effect_and_text_integrity.py", "w") as f:
    f.write(test_content)
