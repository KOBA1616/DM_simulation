#include "mcts_evaluator.hpp"
#include "../inference/native_inference.hpp"
#include <iostream>

using namespace dm::ai::inference;

namespace dm::ai::mcts {

BatchEvaluator::BatchEvaluator(int batch_size) : batch_size_(batch_size) {}

std::vector<float> BatchEvaluator::encode_state_flat(const dm::core::GameState& /*st*/) {
    // Placeholder encoder: return fixed-size zero vector. Replace with real encoder.
    return std::vector<float>(256, 0.0f);
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> BatchEvaluator::evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
    std::pair<std::vector<std::vector<float>>, std::vector<float>> result;
    size_t n = states.size();
    if (n == 0) return result;

    std::vector<std::vector<float>> inputs;
    inputs.reserve(n);
    size_t input_size = 0;
    for (const auto& s : states) {
        auto flat = encode_state_flat(*s);
        if (input_size == 0) input_size = flat.size();
        if (flat.size() != input_size) flat.resize(input_size);
        inputs.push_back(std::move(flat));
    }

    std::vector<float> flat_buf;
    flat_buf.reserve(n * input_size);
    for (size_t i = 0; i < n; ++i) {
        flat_buf.insert(flat_buf.end(), inputs[i].begin(), inputs[i].end());
    }

    auto native_out = NativeInferenceManager::instance().infer_flat_ptr(flat_buf.data(), flat_buf.size(), static_cast<int>(n), static_cast<int>(input_size));

    if (!native_out.second.empty()) {
        result.second = std::move(native_out.second);
    } else {
        result.second.assign(n, 0.0f);
    }

    if (!native_out.first.empty() && n > 0) {
        size_t action_size = native_out.first.size() / n;
        result.first.resize(n);
        for (size_t i = 0; i < n; ++i) {
            result.first[i].assign(native_out.first.begin() + i * action_size, native_out.first.begin() + (i + 1) * action_size);
        }
    } else {
        result.first.resize(n);
    }

    return result;
}

} // namespace dm::ai::mcts
