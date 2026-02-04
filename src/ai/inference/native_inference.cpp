#include "ai/inference/native_inference.hpp"
#include <iostream>

#ifdef USE_ONNXRUNTIME
#include "ai/inference/onnx_model.hpp"
using OnnxModelImpl = dm::ai::inference::OnnxModel;
#endif

#ifdef USE_LIBTORCH
#include "ai/inference/torch_model.hpp"
using TorchModelImpl = dm::ai::TorchModel;
#endif

namespace dm::ai::inference {

NativeInferenceManager& NativeInferenceManager::instance() {
    static NativeInferenceManager inst;
    return inst;
}

bool NativeInferenceManager::load_onnx(const std::string& path) {
#ifdef USE_ONNXRUNTIME
    std::lock_guard<std::mutex> lk(m_);
    try {
        onnx_model_.reset(new OnnxModelImpl(path));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "NativeInferenceManager: failed to load ONNX model: " << e.what() << std::endl;
        onnx_model_.reset();
        return false;
    }
#else
    (void)path;
    return false;
#endif
}

bool NativeInferenceManager::load_torch(const std::string& path) {
#ifdef USE_LIBTORCH
    std::lock_guard<std::mutex> lk(m_);
    try {
        torch_model_.reset(new TorchModelImpl());
        if (!torch_model_->load(path)) {
            torch_model_.reset();
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "NativeInferenceManager: failed to load Torch model: " << e.what() << std::endl;
        torch_model_.reset();
        return false;
    }
#else
    (void)path;
    return false;
#endif
}

void NativeInferenceManager::clear_models() {
    std::lock_guard<std::mutex> lk(m_);
#ifdef USE_ONNXRUNTIME
    onnx_model_.reset();
#endif
#ifdef USE_LIBTORCH
    torch_model_.reset();
#endif
}

std::pair<std::vector<float>, std::vector<float>> NativeInferenceManager::infer_flat(const std::vector<float>& flat, int batch_size, int stride) {
    std::lock_guard<std::mutex> lk(m_);
#ifdef USE_ONNXRUNTIME
    if (onnx_model_) {
        try {
            return onnx_model_->infer_batch(flat, batch_size, stride);
        } catch (const std::exception& e) {
            std::cerr << "NativeInferenceManager: ONNX infer_flat error: " << e.what() << std::endl;
        }
    }
#endif
#ifdef USE_LIBTORCH
    if (torch_model_) {
        try {
            return torch_model_->predict(flat);
        } catch (const std::exception& e) {
            std::cerr << "NativeInferenceManager: Torch predict error: " << e.what() << std::endl;
        }
    }
#endif
    return {{}, {}};
}

std::pair<std::vector<float>, std::vector<float>> NativeInferenceManager::infer_flat_ptr(const float* data, size_t len, int batch_size, int stride) {
    std::lock_guard<std::mutex> lk(m_);
#ifdef USE_ONNXRUNTIME
    if (onnx_model_) {
        try {
            // Build InputTensor referencing external memory (zero-copy)
            InputTensor input;
            const auto &names = onnx_model_->get_input_names();
            if (!names.empty()) input.name = names[0];
            else input.name = "input";
            input.shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(stride)};
            input.data = static_cast<const void*>(data);
            input.type = TensorType::FLOAT;
            return onnx_model_->infer({input}, batch_size);
        } catch (const std::exception& e) {
            std::cerr << "NativeInferenceManager: ONNX infer_flat_ptr error: " << e.what() << std::endl;
        }
    }
#endif
#ifdef USE_LIBTORCH
    if (torch_model_) {
        try {
            // TorchModel currently expects std::vector<float>, copy as fallback
            std::vector<float> buf;
            buf.assign(data, data + len);
            return torch_model_->predict(buf);
        } catch (const std::exception& e) {
            std::cerr << "NativeInferenceManager: Torch predict error: " << e.what() << std::endl;
        }
    }
#endif
    return {{}, {}};
}

bool NativeInferenceManager::has_model() const {
    std::lock_guard<std::mutex> lk(m_);
#ifdef USE_ONNXRUNTIME
    if (onnx_model_) return true;
#endif
#ifdef USE_LIBTORCH
    if (torch_model_) return true;
#endif
    return false;
}

std::pair<std::vector<float>, std::vector<float>> NativeInferenceManager::infer_sequence(const std::vector<std::vector<int>>& tokens) {
    std::lock_guard<std::mutex> lk(m_);
#ifdef USE_ONNXRUNTIME
    if (onnx_model_) {
        // Determine max length
        size_t n = tokens.size();
        size_t max_len = 0;
        for (const auto &t : tokens) if (t.size() > max_len) max_len = t.size();
        if (max_len == 0) return {{}, {}};

        std::vector<int64_t> input_ids;
        std::vector<uint8_t> mask;
        input_ids.reserve(n * max_len);
        mask.reserve(n * max_len);

        for (const auto &seq : tokens) {
            for (size_t i = 0; i < max_len; ++i) {
                if (i < seq.size()) {
                    input_ids.push_back(static_cast<int64_t>(seq[i]));
                    mask.push_back(1);
                } else {
                    input_ids.push_back(0);
                    mask.push_back(0);
                }
            }
        }

        // Build InputTensors
        std::vector<InputTensor> inputs;
        inputs.push_back({"input_ids", {static_cast<int64_t>(n), static_cast<int64_t>(max_len)}, input_ids.data(), TensorType::INT64});
        inputs.push_back({"mask", {static_cast<int64_t>(n), static_cast<int64_t>(max_len)}, mask.data(), TensorType::BOOL});

        try {
            return onnx_model_->infer(inputs, static_cast<int>(n));
        } catch (const std::exception& e) {
            std::cerr << "NativeInferenceManager: ONNX infer_sequence error: " << e.what() << std::endl;
        }
    }
#endif
    // Torch sequence support is not implemented in this manager currently.
    return {{}, {}};
}

} // namespace
