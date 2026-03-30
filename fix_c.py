import os
import re

# 1. card_def.hpp
with open("src/core/card_def.hpp", "r") as f:
    c = f.read()
c = re.sub(r'        // Reaction Abilities\n        std::vector<ReactionAbility> reaction_abilities;\n\n', '', c)
with open("src/core/card_def.hpp", "w") as f:
    f.write(c)

# 2. card_json_types.hpp
with open("src/core/card_json_types.hpp", "r") as f:
    c = f.read()
c = re.sub(r'    struct ReactionCondition \{.*?    \};\n\n', '', c, flags=re.DOTALL)
c = re.sub(r'    struct ReactionAbility \{.*?    \};\n\n', '', c, flags=re.DOTALL)
c = re.sub(r'        std::vector<ReactionAbility> reaction_abilities;\n', '', c)
c = re.sub(r'            \{"reaction_abilities", c\.reaction_abilities\},\n', '', c)
c = re.sub(r'        if \(j\.contains\("reaction_abilities"\)\) \{.*?        \}\n\n', '', c, flags=re.DOTALL)
c = re.sub(r'    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT\(ReactionCondition,.*?\)\n', '', c)
c = re.sub(r'    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT\(ReactionAbility,.*?\)\n\n', '', c)
with open("src/core/card_json_types.hpp", "w") as f:
    f.write(c)

# 3. card_registry.cpp
with open("src/engine/infrastructure/data/card_registry.cpp", "r") as f:
    c = f.read()
c = c.replace(', "reaction_abilities"', '')
with open("src/engine/infrastructure/data/card_registry.cpp", "w") as f:
    f.write(c)

# 4. json_loader.cpp
with open("src/engine/infrastructure/data/json_loader.cpp", "r") as f:
    c = f.read()
c = c.replace(', "reaction_abilities"', '')
c = re.sub(r'        // Reaction Abilities\n        def\.reaction_abilities = data\.reaction_abilities;\n\n', '', c)
with open("src/engine/infrastructure/data/json_loader.cpp", "w") as f:
    f.write(c)

# 5. reaction_system.hpp
with open("src/engine/systems/effects/reaction_system.hpp", "r") as f:
    c = f.read()
c = c.replace("for (const auto& reaction : def.reaction_abilities)", "for (const auto& effect : def.effects)")
c = c.replace("const dm::core::ReactionAbility& reaction,", "const dm::core::EffectDef& effect,")
c = c.replace("reaction.condition.trigger_event", 'effect.condition.str_val')
c = c.replace("reaction.zone", 'effect.condition.stat_key')
c = c.replace("reaction.condition.civilization_match", '(effect.condition.type == "CIVILIZATION_MATCH")')
c = c.replace("reaction.condition.mana_count_min", 'effect.condition.value')
with open("src/engine/systems/effects/reaction_system.hpp", "w") as f:
    f.write(c)

# 6. pending_strategy.cpp
with open("src/engine/command_generation/strategies/pending_strategy.cpp", "r") as f:
    c = f.read()
c = c.replace("for (const auto& r : def.reaction_abilities)", "for (const auto& r : def.effects)")
with open("src/engine/command_generation/strategies/pending_strategy.cpp", "w") as f:
    f.write(c)
