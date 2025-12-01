#include "neural_evaluator.hpp"
#include "../encoders/tensor_converter.hpp"
#include "../../python/python_batch_inference.hpp"
#include <stdexcept>

namespace dm::ai {

    NeuralEvaluator::NeuralEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(card_db) {}

    std::pair<std::vector<std::vector<float>>, std::vector<float>> NeuralEvaluator::evaluate(const std::vector<dm::core::GameState>& states) {
        using BatchInput = dm::python::BatchInput;
        using BatchOutput = dm::python::BatchOutput;

        if (states.empty()) return {{}, {}};

        // Convert states to flat feature vectors using TensorConverter
        const int stride = dm::ai::TensorConverter::INPUT_SIZE;
        std::vector<float> flat = dm::ai::TensorConverter::convert_batch_flat(states, card_db_);

        size_t n = states.size();
        if (flat.size() != n * (size_t)stride) {
            // Fallback: return zeros
            std::vector<std::vector<float>> policies(n, std::vector<float>());
            std::vector<float> values(n, 0.0f);
            return {policies, values};
        }

        // Prefer flat callback (zero-copy path in Python binding) if registered
        if (dm::python::has_flat_batch_callback()) {
            try {
                fprintf(stderr, "NeuralEvaluator: calling flat batch callback (n=%zu, stride=%d)\n", n, stride);
                BatchOutput out = dm::python::call_flat_batch_callback(flat, n, stride);
                return out;
            } catch (const std::exception& e) {
                // fall through to fallback below
            }
        }

        // Fallback: split into row vectors and call the older callback
        BatchInput batch_in;
        batch_in.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            std::vector<float> v(flat.begin() + i * stride, flat.begin() + (i+1) * stride);
            batch_in.push_back(std::move(v));
        }

        if (!dm::python::has_batch_callback()) {
            std::vector<std::vector<float>> policies(n, std::vector<float>());
            std::vector<float> values(n, 0.0f);
            return {policies, values};
        }

        try {
            BatchOutput out = dm::python::call_batch_callback(batch_in);
            return out;
        } catch (const std::exception& e) {
            // On error, fallback
            std::vector<std::vector<float>> policies(n, std::vector<float>());
            std::vector<float> values(n, 0.0f);
            return {policies, values};
        }
    }

}
