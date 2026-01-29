#pragma once

#include <vector>
#include <mutex>
#include <functional>
#include <Python.h>
#include "core/game_state.hpp"

namespace dm::ai {

    class BatchEvaluator {
        std::vector<dm::core::GameState*> pending_states;
        std::vector<std::function<void(const std::vector<float>&, float)>> pending_callbacks;
        std::mutex queue_mutex;

    public:
        // Enqueue a leaf node state for evaluation.
        // The callback receives the policy logits (vector<float>) and value (float).
        // Using vector reference for safety instead of raw pointer.
        void enqueue(dm::core::GameState* state, std::function<void(const std::vector<float>&, float)> callback);

        // Flush the queue and evaluate using the provided Python model.
        // The model is expected to be a callable (e.g. PyTorch module).
        void flush_and_evaluate(PyObject* model);
    };

}
