#include "python_batch_inference.hpp"
#include <mutex>
#include <stdexcept>
#include <Python.h>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
// No SEH usage here; we log around the callback and handle C++ exceptions.

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

    static void append_log(const std::string &s) {
        try {
            std::ofstream f("crash_diag.log", std::ios::app);
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            std::tm tm;
#ifdef _MSC_VER
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            std::ostringstream ss;
            ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " - " << s << "\n";
            f << ss.str();
        } catch(...) {}
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
        append_log("call_batch_callback: input batch size=" + std::to_string(input.size()));
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            auto out = cb_copy(input);
            PyGILState_Release(_gstate);
            append_log("call_batch_callback: success");
            return out;
        
        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log(std::string("call_batch_callback: C++ exception: ") + e.what());
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log("call_batch_callback: unknown exception");
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
        FlatBatchCallback cb_copy;
        {
            std::lock_guard<std::mutex> lk(g_flat_cb_mutex);
            cb_copy = g_flat_callback;
        }
        if (!cb_copy) {
            throw std::runtime_error("No flat batch inference callback registered");
        }
        append_log("call_flat_batch_callback: flat.size=" + std::to_string(flat.size()) + " n=" + std::to_string(n) + " stride=" + std::to_string(stride));
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            auto out = cb_copy(flat, n, stride);
            PyGILState_Release(_gstate);
            append_log("call_flat_batch_callback: success");
            return out;

        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log(std::string("call_flat_batch_callback: C++ exception: ") + e.what());
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log("call_flat_batch_callback: unknown exception");
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
        SequenceBatchCallback cb_copy;
        {
            std::lock_guard<std::mutex> lk(g_seq_cb_mutex);
            cb_copy = g_seq_callback;
        }
        if (!cb_copy) {
            throw std::runtime_error("No sequence batch inference callback registered");
        }
        append_log("call_sequence_batch_callback: input size=" + std::to_string(input.size()));
        PyGILState_STATE _gstate = PyGILState_Ensure();
        try {
            auto out = cb_copy(input);
            PyGILState_Release(_gstate);
            append_log("call_sequence_batch_callback: success");
            return out;

        } catch (const std::exception &e) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log(std::string("call_sequence_batch_callback: C++ exception: ") + e.what());
            PyGILState_Release(_gstate);
            throw;
        } catch(...) {
            if (PyErr_Occurred()) PyErr_Print();
            append_log("call_sequence_batch_callback: unknown exception");
            PyGILState_Release(_gstate);
            throw;
        }
    }

}
