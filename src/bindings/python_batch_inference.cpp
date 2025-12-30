#include "python_batch_inference.hpp"
#include <mutex>
#include <stdexcept>
#include <Python.h>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>

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
        // Acquire GIL immediately to ensure safe manipulation of Python objects (std::function copy/destruct)
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            BatchCallback cb_copy;
            {
                std::lock_guard<std::mutex> lk(g_cb_mutex);
                cb_copy = g_callback;
            }
            if (!cb_copy) {
                // cb_copy will be destroyed when we leave scope or throw.
                // Since GIL is held, destruction is safe.
                throw std::runtime_error("No batch inference callback registered");
            }

            auto out = cb_copy(input);

            // Explicitly destroy the callback copy while holding GIL to ensure safe Py_DECREF
            cb_copy = nullptr;

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
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            FlatBatchCallback cb_copy;
            {
                std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
                cb_copy = g_flat_callback;
            }
            if (!cb_copy) {
                throw std::runtime_error("No flat batch inference callback registered");
            }

            auto out = cb_copy(flat, n, stride);

            cb_copy = nullptr;

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
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            SequenceBatchCallback cb_copy;
            {
                std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
                cb_copy = g_seq_callback;
            }
            if (!cb_copy) {
                throw std::runtime_error("No sequence batch inference callback registered");
            }

            auto out = cb_copy(input);

            cb_copy = nullptr;

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
