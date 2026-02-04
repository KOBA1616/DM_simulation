#include "python_batch_inference.hpp"
#include <mutex>
#include <stdexcept>
#include <Python.h>
#include <iostream>
// Attempt to use native inference manager if available
#include "ai/inference/native_inference.hpp"

namespace dm::python {

    static BatchCallback g_callback = nullptr;
    static std::mutex g_cb_mutex;
    static FlatBatchCallback g_flat_callback = nullptr;
    static std::mutex g_flat_cb_mutex;
    static SequenceBatchCallback g_seq_callback = nullptr;
    static std::mutex g_seq_cb_mutex;

    void set_batch_callback(BatchCallback cb) {
        std::lock_guard<std::mutex> lk(g_cb_mutex);
        g_callback = std::move(cb);
    }

    bool has_batch_callback() {
        std::lock_guard<std::mutex> lk(g_cb_mutex);
        return (bool)g_callback;
    }

    BatchOutput call_batch_callback(const BatchInput& input) {
        // Ensure GIL is held for the entire duration:
        // 1. Copying g_callback (incref)
        // 2. Calling cb_copy
        // 3. Destroying cb_copy (decref)
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            BatchOutput out;
            {
                BatchCallback cb_copy;
                {
                    std::lock_guard<std::mutex> lk(g_cb_mutex);
                    cb_copy = g_callback;
                }
                if (!cb_copy) {
                    throw std::runtime_error("No batch inference callback registered");
                }
                out = cb_copy(input);
            } // cb_copy is destroyed here, while GIL is held

            PyGILState_Release(_gstate);
            return out;
        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        }
    }

    void set_flat_batch_callback(FlatBatchCallback cb) {
        std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
        g_flat_callback = std::move(cb);
    }

    bool has_flat_batch_callback() {
        std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
        return (bool)g_flat_callback;
    }

    void clear_batch_callback() {
        std::lock_guard<std::mutex> lk(g_cb_mutex);
        g_callback = nullptr;
    }

    void clear_flat_batch_callback() {
        std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
        g_flat_callback = nullptr;
    }

    BatchOutput call_flat_batch_callback(const std::vector<float>& flat, size_t n, size_t stride) {
        // Use native inference if a native model is loaded (checked at startup);
        // do not fall back to Python dynamically inside runtime loops.
        if (dm::ai::inference::NativeInferenceManager::instance().has_model()) {
            auto native_out = dm::ai::inference::NativeInferenceManager::instance().infer_flat_ptr(flat.data(), flat.size(), static_cast<int>(n), static_cast<int>(stride));
            if (!native_out.first.empty() || !native_out.second.empty()) {
                BatchOutput out;
                if (!native_out.first.empty() && n > 0 && (native_out.first.size() % n == 0)) {
                    size_t action_size = native_out.first.size() / n;
                    out.first.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        out.first[i].assign(native_out.first.begin() + i * action_size, native_out.first.begin() + (i + 1) * action_size);
                    }
                }
                out.second = std::move(native_out.second);
                return out;
            }
            throw std::runtime_error("Native inference returned empty outputs");
        }

        // No native model loaded -> call Python callback (only path to use Python).
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            BatchOutput out;
            {
                FlatBatchCallback cb_copy;
                {
                    std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
                    cb_copy = g_flat_callback;
                }
                if (!cb_copy) {
                    PyGILState_Release(_gstate);
                    throw std::runtime_error("No flat batch inference callback registered");
                }
                out = cb_copy(flat, n, stride);
            }
            PyGILState_Release(_gstate);
            return out;
        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        }
    }

    void set_sequence_batch_callback(SequenceBatchCallback cb) {
        std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
        g_seq_callback = std::move(cb);
    }

    bool has_sequence_batch_callback() {
        std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
        return (bool)g_seq_callback;
    }

    void clear_sequence_batch_callback() {
        std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
        g_seq_callback = nullptr;
    }

    BatchOutput call_sequence_batch_callback(const SequenceBatchInput& input) {
        // Prefer native sequence inference if model loaded; do not dynamically fall back.
        if (dm::ai::inference::NativeInferenceManager::instance().has_model()) {
            auto native_out = dm::ai::inference::NativeInferenceManager::instance().infer_sequence(input);
            if (!native_out.first.empty() || !native_out.second.empty()) {
                BatchOutput out;
                if (!native_out.first.empty() && input.size() > 0 && (native_out.first.size() % input.size() == 0)) {
                    size_t action_size = native_out.first.size() / input.size();
                    out.first.resize(input.size());
                    for (size_t i = 0; i < input.size(); ++i) {
                        out.first[i].assign(native_out.first.begin() + i * action_size, native_out.first.begin() + (i + 1) * action_size);
                    }
                }
                out.second = std::move(native_out.second);
                return out;
            }
            throw std::runtime_error("Native sequence inference returned empty outputs");
        }

        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            BatchOutput out;
            {
                SequenceBatchCallback cb_copy;
                {
                    std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
                    cb_copy = g_seq_callback;
                }
                if (!cb_copy) {
                    PyGILState_Release(_gstate);
                    throw std::runtime_error("No sequence batch inference callback registered");
                }
                out = cb_copy(input);
            }
            PyGILState_Release(_gstate);
            return out;
        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            PyGILState_Release(_gstate);
            throw;
        }
    }

}
