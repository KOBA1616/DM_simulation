#pragma once
#include "engine/command_generation/command_strategy.hpp"

namespace dm::engine {

    class StackStrategy : public ICommandStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) override;
    };

}
