#pragma once
#include <vector>
#include <functional>

namespace dm::python {

    using BatchInput = std::vector<std::vector<float>>;
    using BatchOutput = std::pair<std::vector<std::vector<float>>, std::vector<float>>;
    using BatchCallback = std::function<BatchOutput(const BatchInput&)>;

    // Flat (contiguous) batch callback: flat vector, batch size, stride
    using FlatBatchCallback = std::function<BatchOutput(const std::vector<float>& flat, size_t n, size_t stride)>;

    // Sequence (token) batch callback: vector of vector of ints
    using SequenceBatchInput = std::vector<std::vector<int>>;
    using SequenceBatchCallback = std::function<BatchOutput(const SequenceBatchInput&)>;

    // Set the batch callback (thread-safe from caller side)
    void set_batch_callback(BatchCallback cb);

    // Whether a callback is registered
    bool has_batch_callback();

    // Call the registered callback (throws if none)
    BatchOutput call_batch_callback(const BatchInput& input);

    // Flat API: set/get/call when providing a contiguous flat buffer
    void set_flat_batch_callback(FlatBatchCallback cb);
    bool has_flat_batch_callback();

    // Clear registered callbacks
    void clear_batch_callback();
    void clear_flat_batch_callback();
    BatchOutput call_flat_batch_callback(const std::vector<float>& flat, size_t n, size_t stride);

    // Sequence API
    void set_sequence_batch_callback(SequenceBatchCallback cb);
    bool has_sequence_batch_callback();
    void clear_sequence_batch_callback();
    BatchOutput call_sequence_batch_callback(const SequenceBatchInput& input);

}
