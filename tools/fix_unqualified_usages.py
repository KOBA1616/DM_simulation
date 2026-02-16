import os
import re

# Map class names to fully qualified names
mapping = {
    'EffectSystem': 'dm::engine::effects::EffectSystem',
    'CardRegistry': 'dm::engine::infrastructure::CardRegistry',
    'ConditionSystem': 'dm::engine::rules::ConditionSystem',
    'TargetUtils': 'dm::engine::utils::TargetUtils',
    'SelectionSystem': 'dm::engine::mechanics::SelectionSystem',
    'ReactionSystem': 'dm::engine::effects::ReactionSystem',
    'JsonLoader': 'dm::engine::infrastructure::JsonLoader',
    'KeywordExpander': 'dm::engine::infrastructure::KeywordExpander',
}

# Namespaces to skip if we are inside them?
# To be safe, we always fully qualify usage in other files.

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        for class_name, fqn in mapping.items():
            # Skip definition files
            if os.path.basename(filepath) == class_name_to_filename(class_name):
                continue

            # Regex to find usages:
            # Not preceded by "class ", "struct ", "::", "include", or namespace declaration parts
            # And matches word boundary

            # 1. Skip if preceded by "::" (already qualified or part of another name)
            # 2. Skip if preceded by "class " or "struct " (declaration)
            # 3. Skip if in #include
            # 4. Skip if preceded by "namespace " (unlikely for usage)

            pattern = r'(?<!::)(?<!class\s)(?<!struct\s)(?<!namespace\s)(?<!#include\s")(?<!#include\s<)\b' + re.escape(class_name) + r'\b'

            def replace_func(match):
                # Double check we are not inside a string or comment? (Simple script might miss this)
                # But also check if it's already qualified by something else?
                # The negative lookbehind (?<!::) handles "dm::engine::EffectSystem".
                # But what about "effects::EffectSystem"? That has "::".
                return fqn

            content = re.sub(pattern, replace_func, content)

        if content != original_content:
            print(f"Qualifying usages in {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def class_name_to_filename(name):
    # Primitive mapping
    if name == 'EffectSystem': return 'effect_system.hpp'
    if name == 'CardRegistry': return 'card_registry.hpp'
    if name == 'ConditionSystem': return 'condition_system.hpp'
    if name == 'TargetUtils': return 'target_utils.hpp'
    if name == 'SelectionSystem': return 'selection_system.hpp'
    if name == 'ReactionSystem': return 'reaction_system.hpp'
    if name == 'JsonLoader': return 'json_loader.hpp'
    if name == 'KeywordExpander': return 'keyword_expander.hpp'
    return ""

for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith((".hpp", ".cpp")):
            process_file(os.path.join(root, file))
