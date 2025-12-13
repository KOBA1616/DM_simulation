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
        // Create input tensor
        std::vector<int64_t> input_node_dims = {batch_size, input_size};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Cast away constness because ORT API requires non-const pointer,
        // but we are creating a tensor over existing memory which we treat as read-only for input.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_node_dims.data(),
            input_node_dims.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(),
            &input_tensor,
            1,
            output_node_names_.data(),
            output_node_names_.size()
        );

        // Extract outputs
        // Assuming Output 0 is Policy and Output 1 is Value based on export script

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
