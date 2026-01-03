#include "onnx_model.hpp"
#include <iostream>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

namespace dm::ai::inference {

#ifdef USE_ONNXRUNTIME

    OnnxModel::OnnxModel(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "DM_AI_Inference")
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, &model_path[0], (int)model_path.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &model_path[0], (int)model_path.size(), &wstrTo[0], size_needed);
        session_ = std::make_unique<Ort::Session>(env_, wstrTo.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        // Get Input Names
        size_t num_input_nodes = session_->GetInputCount();
        input_node_names_.resize(num_input_nodes);
        input_node_names_str_.resize(num_input_nodes);

        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session_->GetInputNameAllocated(i, *allocator_);
            input_node_names_str_[i] = name.get();
            input_node_names_[i] = input_node_names_str_[i].c_str();
        }

        // Get Output Names
        size_t num_output_nodes = session_->GetOutputCount();
        output_node_names_.resize(num_output_nodes);
        output_node_names_str_.resize(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session_->GetOutputNameAllocated(i, *allocator_);
            output_node_names_str_[i] = name.get();
            output_node_names_[i] = output_node_names_str_[i].c_str();
        }
    }

    OnnxModel::~OnnxModel() = default;

    std::pair<std::vector<float>, std::vector<float>> OnnxModel::infer_batch(
        const std::vector<float>& input_data,
        int batch_size,
        int input_size
    ) {
        // Delegate to generic infer
        InputTensor input;
        // Use the name detected from the model if available, otherwise "input"
        if (!input_node_names_.empty()) {
            input.name = input_node_names_[0];
        } else {
            input.name = "input";
        }
        input.shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(input_size)};
        input.data = input_data.data();
        input.type = TensorType::FLOAT;

        return infer({input}, batch_size);
    }

    std::pair<std::vector<float>, std::vector<float>> OnnxModel::infer(
        const std::vector<InputTensor>& inputs,
        int batch_size
    ) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;
        input_tensors.reserve(inputs.size());
        input_names.reserve(inputs.size());

        for (const auto& inp : inputs) {
            input_names.push_back(inp.name.c_str());

            size_t element_count = 1;
            for(auto d : inp.shape) element_count *= d;

            if (inp.type == TensorType::FLOAT) {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    const_cast<float*>(static_cast<const float*>(inp.data)),
                    element_count,
                    inp.shape.data(),
                    inp.shape.size()
                ));
            } else if (inp.type == TensorType::INT64) {
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info,
                    const_cast<int64_t*>(static_cast<const int64_t*>(inp.data)),
                    element_count,
                    inp.shape.data(),
                    inp.shape.size()
                ));
            } else if (inp.type == TensorType::BOOL) {
                // Ensure bool size safety
                static_assert(sizeof(bool) == 1, "bool must be 1 byte for direct cast from uint8_t");

                // Assuming inp.data points to uint8_t array
                input_tensors.push_back(Ort::Value::CreateTensor<bool>(
                    memory_info,
                    const_cast<bool*>(static_cast<const bool*>(inp.data)),
                    element_count,
                    inp.shape.data(),
                    inp.shape.size()
                ));
            } else {
                 throw std::runtime_error("Unsupported TensorType in OnnxModel::infer");
            }
        }

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names_.data(),
            output_node_names_.size()
        );

        // Extract outputs
        // Assuming Output 0 is Policy and Output 1 is Value based on AlphaZero standard
        // This part is specific to the RL usecase, but the input part is now generic.

        // Safety check
        if (output_tensors.size() < 2) {
             throw std::runtime_error("Model did not return enough outputs (expected policy and value)");
        }

        float* policy_arr = output_tensors[0].GetTensorMutableData<float>();
        size_t policy_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> policy_out(policy_arr, policy_arr + policy_count);

        float* value_arr = output_tensors[1].GetTensorMutableData<float>();
        size_t value_count = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> value_out(value_arr, value_arr + value_count);

        return {policy_out, value_out};
    }

#endif

}
