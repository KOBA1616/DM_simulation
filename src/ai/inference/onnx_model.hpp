#pragma once

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <mutex>

// Conditional compilation for ONNX Runtime
// This allows compiling the rest of the project without ONNX runtime dependencies
#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace dm::ai::inference {

    enum class TensorType {
        FLOAT,
        INT64,
        BOOL
    };

    struct InputTensor {
        std::string name;
        std::vector<int64_t> shape;
        const void* data;
        TensorType type;
    };

    class OnnxModel {
    public:
#ifdef USE_ONNXRUNTIME
        OnnxModel(const std::string& model_path);
        ~OnnxModel();

        // Batch inference (Legacy/Convenience wrapper)
        // Input: flattened vector of features (batch_size * input_size)
        // Output: pair of (policies, values)
        std::pair<std::vector<float>, std::vector<float>> infer_batch(
            const std::vector<float>& input_data,
            int batch_size,
            int input_size
        );

        // Generic inference
        // Input: List of named input tensors
        // Output: pair of (policies, values) - assumes standard output format for now
        // Note: For full generic support, return type should also be map of outputs,
        // but current usage expects {policy, value}.
        std::pair<std::vector<float>, std::vector<float>> infer(
            const std::vector<InputTensor>& inputs,
            int batch_size
        );

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

        std::vector<std::string> input_node_names_str_;
        std::vector<std::string> output_node_names_str_;
        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;
#else
        // Dummy implementation when ONNX Runtime is not available
        OnnxModel(const std::string& /*model_path*/) {}
        ~OnnxModel() {}
        std::pair<std::vector<float>, std::vector<float>> infer_batch(
            const std::vector<float>& /*input_data*/,
            int /*batch_size*/,
            int /*input_size*/
        ) {
            return {{}, {}};
        }

        std::pair<std::vector<float>, std::vector<float>> infer(
            const std::vector<InputTensor>& /*inputs*/,
            int /*batch_size*/
        ) {
            return {{}, {}};
        }
#endif
    };

}
