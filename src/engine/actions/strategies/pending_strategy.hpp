#pragma once
#include "engine/actions/action_strategy.hpp"

namespace dm::engine {

    class PendingEffectStrategy : public IActionStrategy {
    public:
        std::vector<dm::core::CommandDef> generate(const ActionGenContext& ctx) override;
    };

}
