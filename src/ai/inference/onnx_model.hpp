#pragma once

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

namespace dm::ai::inference {

    class OnnxModel {
    public:
        OnnxModel(const std::string& model_path);
        ~OnnxModel();

        // Batch inference
        // Input: flattened vector of features (batch_size * input_size)
        // Output: pair of (policies, values)
        // policies: flattened (batch_size * action_size)
        // values: flattened (batch_size * 1)
        std::pair<std::vector<float>, std::vector<float>> infer_batch(
            const std::vector<float>& input_data,
            int batch_size,
            int input_size
        );

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

        std::vector<std::string> input_node_names_str_;
        std::vector<std::string> output_node_names_str_;
        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;
    };

}
#endif
