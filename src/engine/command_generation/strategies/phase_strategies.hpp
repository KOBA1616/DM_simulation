#pragma once
#include "engine/command_generation/command_strategy.hpp"

namespace dm::engine {

    class ManaPhaseStrategy : public ICommandStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) override;
    };

    class MainPhaseStrategy : public ICommandStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) override;
    };

    class AttackPhaseStrategy : public ICommandStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) override;
    };

    class BlockPhaseStrategy : public ICommandStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const CommandGenContext& ctx) override;
    };

}
