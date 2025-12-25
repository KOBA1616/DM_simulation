#include "neural_evaluator.hpp"
#include "ai/encoders/tensor_converter.hpp"
#include "ai/encoders/token_converter.hpp"
#include "bindings/python_batch_inference.hpp"
#include <stdexcept>
#include <iostream>

#ifdef USE_ONNXRUNTIME
#include "ai/inference/onnx_model.hpp"
#endif

namespace dm::ai {

    NeuralEvaluator::NeuralEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(card_db) {}

    NeuralEvaluator::~NeuralEvaluator() = default;

    void NeuralEvaluator::load_model(const std::string& path) {
#ifdef USE_ONNXRUNTIME
        try {
            onnx_model_ = std::make_unique<dm::ai::inference::OnnxModel>(path);
            std::cout << "Loaded ONNX model from " << path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
            onnx_model_.reset();
        }
#else
        (void)path;
        std::cerr << "Warning: ONNX Runtime support is not compiled in." << std::endl;
#endif
    }

    void NeuralEvaluator::set_model_type(ModelType type) {
        model_type_ = type;
    }

    std::pair<std::vector<std::vector<float>>, std::vector<float>> NeuralEvaluator::evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
        using BatchInput = dm::python::BatchInput;
        using BatchOutput = dm::python::BatchOutput;

        if (states.empty()) return {{}, {}};
        size_t n = states.size();

        // 1. Transformer (Sequence) Path
        if (model_type_ == ModelType::TRANSFORMER) {
#ifdef USE_ONNXRUNTIME
            if (onnx_model_) {
                try {
                    // 1. Collect tokens and find max length
                    std::vector<std::vector<int>> batch_tokens;
                    batch_tokens.reserve(n);
                    size_t max_len = 0;
                    for (const auto& s : states) {
                        if (!s) {
                            batch_tokens.push_back({});
                            continue;
                        }
                        auto tokens = dm::ai::encoders::TokenConverter::encode_state(*s, s->active_player_id);
                        if (tokens.size() > max_len) max_len = tokens.size();
                        batch_tokens.push_back(std::move(tokens));
                    }

                    // 2. Prepare flat input tensors (input_ids, mask)
                    std::vector<int64_t> input_ids;
                    std::vector<uint8_t> mask; // Use uint8_t instead of bool to ensure .data() exists and is contiguous
                    input_ids.reserve(n * max_len);
                    mask.reserve(n * max_len);

                    for(const auto& tokens : batch_tokens) {
                        // Pad to max_len
                        for(size_t i=0; i<max_len; ++i) {
                            if (i < tokens.size()) {
                                input_ids.push_back(static_cast<int64_t>(tokens[i]));
                                mask.push_back(1);
                            } else {
                                input_ids.push_back(static_cast<int64_t>(dm::ai::encoders::TokenConverter::TOKEN_PAD));
                                mask.push_back(0);
                            }
                        }
                    }

                    // 3. Construct InputTensor list
                    const auto& input_names = onnx_model_->get_input_names();
                    std::string input_ids_name = "input_ids";
                    std::string mask_name = "mask";

                    // Try to match detected names if possible
                    if (input_names.size() >= 2) {
                        // Heuristic: usually first is input_ids, second is mask, or by name
                        // For now we trust the export script names or standard names.
                        // Ideally we should check input_names contains "mask" etc.
                        for(const auto& name : input_names) {
                            if (name.find("mask") != std::string::npos) mask_name = name;
                            if (name.find("input") != std::string::npos || name.find("ids") != std::string::npos) input_ids_name = name;
                        }
                    }

                    std::vector<dm::ai::inference::InputTensor> inputs;
                    inputs.push_back({input_ids_name, {static_cast<int64_t>(n), static_cast<int64_t>(max_len)}, input_ids.data(), dm::ai::inference::TensorType::INT64});
                    inputs.push_back({mask_name, {static_cast<int64_t>(n), static_cast<int64_t>(max_len)}, mask.data(), dm::ai::inference::TensorType::BOOL});

                    // 4. Infer
                    auto result = onnx_model_->infer(inputs, n);
                    const auto& flat_policies = result.first;
                    const auto& flat_values = result.second;

                    // 5. Reshape Output
                    std::vector<std::vector<float>> policies(n);
                    std::vector<float> values = flat_values;

                    size_t action_size = 0;
                    if (n > 0 && !flat_policies.empty()) {
                        action_size = flat_policies.size() / n;
                    }

                    for(size_t i=0; i<n; ++i) {
                        policies[i].assign(flat_policies.begin() + i*action_size, flat_policies.begin() + (i+1)*action_size);
                    }

                    return {policies, values};

                } catch (const std::exception& e) {
                    std::cerr << "ONNX Inference Error (Transformer): " << e.what() << std::endl;
                    // Fallthrough to python callback
                }
            }
#endif

            if (dm::python::has_sequence_batch_callback()) {
                dm::python::SequenceBatchInput batch_tokens;
                batch_tokens.reserve(n);
                for (const auto& s : states) {
                    if (!s) {
                        batch_tokens.push_back({});
                        continue;
                    }
                    // Use active player as perspective, max_len=0 for now (or dynamic)
                    batch_tokens.push_back(dm::ai::encoders::TokenConverter::encode_state(*s, s->active_player_id));
                }

                try {
                    return dm::python::call_sequence_batch_callback(batch_tokens);
                } catch (const std::exception& e) {
                    std::cerr << "NeuralEvaluator: Sequence callback failed: " << e.what() << std::endl;
                    // Fallback to zeros
                }
            } else {
                std::cerr << "NeuralEvaluator: Transformer mode selected but no sequence callback registered." << std::endl;
            }

            // Fallback for missing callback
            std::vector<std::vector<float>> policies(n, std::vector<float>());
            std::vector<float> values(n, 0.0f);
            return {policies, values};
        }

        // 2. ResNet (Flat Tensor) Path

        // Convert states to flat feature vectors using TensorConverter
        const int stride = dm::ai::TensorConverter::INPUT_SIZE;
        std::vector<float> flat = dm::ai::TensorConverter::convert_batch_flat(states, card_db_);

        if (flat.size() != n * (size_t)stride) {
            // Fallback: return zeros
            std::vector<std::vector<float>> policies(n, std::vector<float>());
            std::vector<float> values(n, 0.0f);
            return {policies, values};
        }

#ifdef USE_ONNXRUNTIME
        if (onnx_model_) {
            try {
                auto result = onnx_model_->infer_batch(flat, n, stride);
                const auto& flat_policies = result.first;
                const auto& flat_values = result.second;

                std::vector<std::vector<float>> policies(n);
                std::vector<float> values = flat_values; // Copy values (which are n * 1 usually)

                size_t action_size = 0;
                if (n > 0 && !flat_policies.empty()) {
                    action_size = flat_policies.size() / n;
                }

                for(size_t i=0; i<n; ++i) {
                    policies[i].assign(flat_policies.begin() + i*action_size, flat_policies.begin() + (i+1)*action_size);
                }

                return {policies, values};
            } catch (const std::exception& e) {
                std::cerr << "ONNX Inference Error: " << e.what() << std::endl;
                // Fallthrough to python fallback
            }
        }
#endif

        // Prefer flat callback (zero-copy path in Python binding) if registered
        if (dm::python::has_flat_batch_callback()) {
            try {
                #ifdef AI_DEBUG
                fprintf(stderr, "NeuralEvaluator: calling flat batch callback (n=%zu, stride=%d)\n", n, stride);
                #endif
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
