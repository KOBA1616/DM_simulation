#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mutex>

namespace dm::ai::inference {

#ifdef USE_ONNXRUNTIME
    class OnnxModel; // forward decl (namespace scope)
#endif

#ifdef USE_LIBTORCH
    class TorchModel; // forward decl (namespace scope)
#endif

    class NativeInferenceManager {
    public:
        static NativeInferenceManager& instance();

        // Load model backends. Returns true on success.
        bool load_onnx(const std::string& path);
        bool load_torch(const std::string& path);

        // Clear loaded models
        void clear_models();

        // Flat batch inference: returns pair(policy_flat, values_flat)
        std::pair<std::vector<float>, std::vector<float>> infer_flat(const std::vector<float>& flat, int batch_size, int stride);

        // Zero-copy inference entry that accepts a raw float pointer and length.
        // Bindings can forward numpy memory (py::array_t) directly into this call
        // to avoid an intermediate std::vector copy on the hot path.
        std::pair<std::vector<float>, std::vector<float>> infer_flat_ptr(const float* data, size_t len, int batch_size, int stride);

        // Sequence (token) inference: tokens is list of token sequences
        // Returns pair(policies_per_batch_flattened, values_flat)
        std::pair<std::vector<float>, std::vector<float>> infer_sequence(const std::vector<std::vector<int>>& tokens);

        // Return true if any native model backend is loaded
        bool has_model() const;

    private:
        NativeInferenceManager() = default;
        ~NativeInferenceManager() = default;
        NativeInferenceManager(const NativeInferenceManager&) = delete;
        NativeInferenceManager& operator=(const NativeInferenceManager&) = delete;

        mutable std::mutex m_;

    #ifdef USE_ONNXRUNTIME
        std::unique_ptr<OnnxModel> onnx_model_;
    #endif

    #ifdef USE_LIBTORCH
        std::unique_ptr<TorchModel> torch_model_;
    #endif
    };

} // namespace
