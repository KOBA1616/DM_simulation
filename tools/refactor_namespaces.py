import os

replacements = [
    ('#include "engine/systems/card/card_registry.hpp"', '#include "engine/infrastructure/data/card_registry.hpp"'),
    ('#include "engine/systems/card/json_loader.hpp"', '#include "engine/infrastructure/data/json_loader.hpp"'),
    ('#include "engine/systems/card/keyword_expander.hpp"', '#include "engine/infrastructure/data/keyword_expander.hpp"'),
    ('#include "engine/systems/card/condition_system.hpp"', '#include "engine/systems/rules/condition_system.hpp"'),
    ('#include "engine/systems/card/target_utils.hpp"', '#include "engine/utils/target_utils.hpp"'),
    ('#include "engine/systems/card/effect_system.hpp"', '#include "engine/systems/effects/effect_system.hpp"'),
    ('#include "engine/systems/card/selection_system.hpp"', '#include "engine/systems/mechanics/selection_system.hpp"'),
    ('#include "engine/systems/flow/reaction_system.hpp"', '#include "engine/systems/effects/reaction_system.hpp"'),
    ('dm::engine::CardRegistry', 'dm::engine::infrastructure::CardRegistry'),
    ('dm::engine::JsonLoader', 'dm::engine::infrastructure::JsonLoader'),
    ('dm::engine::KeywordExpander', 'dm::engine::infrastructure::KeywordExpander'),
    ('dm::engine::ConditionSystem', 'dm::engine::rules::ConditionSystem'),
    ('dm::engine::TargetUtils', 'dm::engine::utils::TargetUtils'),
    ('dm::engine::EffectSystem', 'dm::engine::effects::EffectSystem'),
    ('dm::engine::SelectionSystem', 'dm::engine::mechanics::SelectionSystem'),
    ('dm::engine::ReactionSystem', 'dm::engine::effects::ReactionSystem'),
]

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        for search, replace in replacements:
            content = content.replace(search, replace)

        if content != original_content:
            print(f"Updating {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith((".hpp", ".cpp")):
            process_file(os.path.join(root, file))
