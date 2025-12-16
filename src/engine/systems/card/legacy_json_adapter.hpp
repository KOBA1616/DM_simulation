#pragma once
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include <vector>

namespace dm::engine::systems {

    /**
     * @brief Adapts legacy EffectDef/ActionDef structures to the new Instruction Pipeline format.
     *
     * This class facilitates the Phase 6 migration by allowing existing JSON card data
     * to be executed by the new PipelineExecutor without manual rewriting of thousands of cards.
     */
    class LegacyJsonAdapter {
    public:
        /**
         * @brief Converts a legacy EffectDef into a sequence of Instructions.
         * @param effect The legacy effect definition.
         * @return A vector of Instructions ready for PipelineExecutor.
         */
        static std::vector<core::Instruction> convert(const core::EffectDef& effect);

    private:
        static void convert_action(const core::ActionDef& action, std::vector<core::Instruction>& out);
    };

}
