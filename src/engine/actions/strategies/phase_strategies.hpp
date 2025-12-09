#pragma once
#include "../action_strategy.hpp"

namespace dm::engine {

    class ManaPhaseStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::Action> generate(const ActionGenContext& ctx) override;
    };

    class MainPhaseStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::Action> generate(const ActionGenContext& ctx) override;
    };

    class AttackPhaseStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::Action> generate(const ActionGenContext& ctx) override;
    };

    class BlockPhaseStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::Action> generate(const ActionGenContext& ctx) override;
    };

}
