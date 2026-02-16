# Native implementation signatures (proposal)

Purpose: Provide C++ header-style signatures and notes for implementing native versions of key components so Python shims can be replaced by native bindings.

## DataCollector (pybind-friendly)

```cpp
// DataCollector.h
#pragma once

#include <vector>
#include <string>

struct DataSample {
    // minimal fields - extend as needed
    std::vector<float> features;
    std::vector<int> tokens;
    int label;
};

struct DataBatch {
    std::vector<DataSample> values;
};

class DataCollector {
public:
    // Optionally accept a card database pointer/reference
    DataCollector(const CardDatabase* card_db = nullptr) noexcept;

    // Heuristic batch collection used by Python tests
    // Returns a DataBatch containing up to batch_size samples.
    DataBatch collect_data_batch_heuristic(int batch_size, bool include_history, bool include_features);

    // Optional: streaming API to fill preallocated buffers (for zero-copy into numpy)
    void collect_into_buffers(int batch_size, float* features_out, int* tokens_out, int* offsets_out);
};
```

Notes:
- Expose via pybind11 with a method returning a simple struct convertible to Python dict/list, or provide a `collect_into_buffers` zero-copy API.

## ParallelRunner / Inference bridge

```cpp
// ParallelRunner.h
#pragma once

#include <vector>

class ParallelRunner {
public:
    ParallelRunner(const CardDatabase* card_db, int sims, int batch_size) noexcept;

    struct PlayResult {
        int result_code; // e.g. GameResult enum
        int winner;
        bool is_over;
    };

    // Plays a batch of games given initial states (serialized or state objects exposed to C++)
    std::vector<PlayResult> play_games(const std::vector<GameState>& initial_states,
                                       float temperature,
                                       bool add_noise,
                                       int threads = 1);

    // Optional lifecycle hooks used by Python tests
    void register_batch_inference_numpy(void* input_ptr, void* output_ptr);
    void shutdown();
};
```

Notes:
- `GameState` should have a C++ representation or a compact serialized form to pass between Python and C++.
- `register_batch_inference_numpy` is suggested for advanced zero-copy numpy integrations — implement as needed with pybind11.

## Binding recommendations
- Use pybind11 for bindings; keep function names identical to Python shim for seamless replacement.
- Prefer returning simple POD structs (vectors, ints, floats) which map naturally to Python lists/dicts.
- Provide both convenience high-level methods and low-level buffer APIs for perf-critical paths.

## Next steps for C++ implementers
1. Implement `DataCollector` and `ParallelRunner` headers in the C++ project and add pybind11 wrappers.
2. Run `scripts/check_native_symbols.py` periodically to reduce shim surface.
3. Add unit tests in C++ (if applicable) mirroring the Python tests to verify behavior.


