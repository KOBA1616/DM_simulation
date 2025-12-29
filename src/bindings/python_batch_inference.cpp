#include "python_batch_inference.hpp"
#include <mutex>
#include <stdexcept>
#include <pybind11/pybind11.h>

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
        BatchCallback cb_copy;
        {
            std::lock_guard<std::mutex> lk(g_cb_mutex);
            cb_copy = g_callback;
        }
        if (!cb_copy) {
            throw std::runtime_error("No batch inference callback registered");
        }
        pybind11::gil_scoped_acquire acquire;
        return cb_copy(input);
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
        FlatBatchCallback cb_copy;
        {
            std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
            cb_copy = g_flat_callback;
        }
        if (!cb_copy) {
            throw std::runtime_error("No flat batch inference callback registered");
        }
        pybind11::gil_scoped_acquire acquire;
        return cb_copy(flat, n, stride);
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
        SequenceBatchCallback cb_copy;
        {
            std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
            cb_copy = g_seq_callback;
        }
        if (!cb_copy) {
            throw std::runtime_error("No sequence batch inference callback registered");
        }
        pybind11::gil_scoped_acquire acquire;
        return cb_copy(input);
    }

}
