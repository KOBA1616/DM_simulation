#pragma once
#include "engine/actions/action_strategy.hpp"

namespace dm::engine {

    class StackStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::Action> generate(const ActionGenContext& ctx) override;
    };

}
