#pragma once
#include "evaluator.hpp"
#include <map>
#include <memory>
#include <string>
#include "core/card_def.hpp"

// Forward declaration
namespace dm::ai::inference {
    class OnnxModel;
}

namespace dm::ai {

    enum class ModelType {
        RESNET,
        TRANSFORMER
    };

    class NeuralEvaluator : public IEvaluator {
    public:
        NeuralEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        ~NeuralEvaluator();

        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) override;

        // Load ONNX model from file
        void load_model(const std::string& path);

        // Set model type
        void set_model_type(ModelType type);

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        std::unique_ptr<dm::ai::inference::OnnxModel> onnx_model_;
        ModelType model_type_ = ModelType::RESNET;
    };

}
