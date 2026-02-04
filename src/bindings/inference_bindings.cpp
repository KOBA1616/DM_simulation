#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ai/inference/onnx_model.hpp"
#include "ai/inference/torch_model.hpp"
#include "ai/inference/native_inference.hpp"

namespace py = pybind11;

// Register inference-related bindings into the provided module.
// This function is intended to be called from the single PYBIND11_MODULE
// entry point located in src/bindings/bindings.cpp so bindings can be
// split across multiple translation units.
void bind_inference(py::module &m) {
#ifdef USE_ONNXRUNTIME
    py::class_<dm::ai::inference::OnnxModel>(m, "OnnxModel")
        .def(py::init<const std::string&>())
        .def("infer_batch", [](dm::ai::inference::OnnxModel &self, py::array_t<float> flat, int batch_size, int input_size) {
            auto buf = flat.request();
            if (buf.ndim != 1) throw std::runtime_error("input must be a 1-D float array (flat)");
            size_t expected = static_cast<size_t>(batch_size) * static_cast<size_t>(input_size);
            if (static_cast<size_t>(buf.size) != expected) throw std::runtime_error("input size mismatch for given batch and input_size");
            const float* data = static_cast<const float*>(buf.ptr);
            std::vector<float> in(data, data + buf.size);
            auto out = self.infer_batch(in, batch_size, input_size);

            // Copy results into numpy arrays
            py::array_t<float> policy(out.first.size());
            py::array_t<float> value(out.second.size());
            if (!out.first.empty()) std::copy(out.first.begin(), out.first.end(), policy.mutable_data());
            if (!out.second.empty()) std::copy(out.second.begin(), out.second.end(), value.mutable_data());
            return py::make_tuple(policy, value);
        });
#endif

#ifdef USE_LIBTORCH
    py::class_<dm::ai::TorchModel>(m, "TorchModel")
        .def(py::init<>())
        .def("load", &dm::ai::TorchModel::load)
        .def("predict", [](dm::ai::TorchModel &self, py::array_t<float> flat) {
            auto buf = flat.request();
            if (buf.ndim != 1) throw std::runtime_error("input must be a 1-D float array (flat)");
            const float* data = static_cast<const float*>(buf.ptr);
            std::vector<float> in(data, data + buf.size);
            auto out = self.predict(in);
            py::array_t<float> policy(out.first.size());
            py::array_t<float> value(out.second.size());
            if (!out.first.empty()) std::copy(out.first.begin(), out.first.end(), policy.mutable_data());
            if (!out.second.empty()) std::copy(out.second.begin(), out.second.end(), value.mutable_data());
            return py::make_tuple(policy, value);
        });
#endif

    // Native manager helpers
    m.def("native_load_onnx", [](const std::string &p) {
#ifdef USE_ONNXRUNTIME
        return dm::ai::inference::NativeInferenceManager::instance().load_onnx(p);
#else
        (void)p; return false;
#endif
    }, "Load ONNX model into native manager");

    m.def("native_load_torch", [](const std::string &p) {
#ifdef USE_LIBTORCH
        return dm::ai::inference::NativeInferenceManager::instance().load_torch(p);
#else
        (void)p; return false;
#endif
    }, "Load Torch model into native manager");

    m.def("native_clear", []() {
        dm::ai::inference::NativeInferenceManager::instance().clear_models();
    }, "Clear native models");

    m.def("native_infer_flat", [](py::array_t<float, py::array::c_style | py::array::forcecast> flat, int batch_size, int stride) {
        auto buf = flat.request();
        if (buf.ndim != 1) throw std::runtime_error("native_infer_flat expects 1-D float array");
        const float* data = static_cast<const float*>(buf.ptr);
        size_t len = static_cast<size_t>(buf.size);
        auto out = dm::ai::inference::NativeInferenceManager::instance().infer_flat_ptr(data, len, batch_size, stride);
        py::array_t<float> policy(out.first.size());
        py::array_t<float> values(out.second.size());
        if (!out.first.empty()) std::copy(out.first.begin(), out.first.end(), policy.mutable_data());
        if (!out.second.empty()) std::copy(out.second.begin(), out.second.end(), values.mutable_data());
        return py::make_tuple(policy, values);
    }, "Run flat inference via native manager");

    m.def("native_infer_sequence", [](py::list seqs) {
        std::vector<std::vector<int>> tokens;
        tokens.reserve(seqs.size());
        for (auto item : seqs) {
            std::vector<int> cur;
            for (auto sub : item.cast<py::list>()) {
                cur.push_back(sub.cast<int>());
            }
            tokens.push_back(std::move(cur));
        }
        auto out = dm::ai::inference::NativeInferenceManager::instance().infer_sequence(tokens);
        // Split policies into per-batch vectors if possible
        py::list policies;
        py::array_t<float> values(out.second.size());
        if (!out.second.empty()) std::copy(out.second.begin(), out.second.end(), values.mutable_data());
        if (!out.first.empty()) {
            // Unknown action size per batch; return flat vector for now as single array
            py::array_t<float> flat_policy(out.first.size());
            std::copy(out.first.begin(), out.first.end(), flat_policy.mutable_data());
            policies.append(flat_policy);
        }
        return py::make_tuple(policies, values);
    }, "Run sequence inference via native manager");

}
