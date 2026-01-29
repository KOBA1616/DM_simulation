#include "batch_evaluator.hpp"
#include "encoders/token_converter.hpp"
#include "core/types.hpp"
#include <iostream>

namespace dm::ai {

    void BatchEvaluator::enqueue(dm::core::GameState* state, std::function<void(const std::vector<float>&, float)> callback) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        pending_states.push_back(state);
        pending_callbacks.push_back(std::move(callback));
    }

    void BatchEvaluator::flush_and_evaluate(PyObject* model) {
        std::vector<dm::core::GameState*> states;
        std::vector<std::function<void(const std::vector<float>&, float)>> callbacks;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (pending_states.empty()) return;
            states.swap(pending_states);
            callbacks.swap(pending_callbacks);
        }

        // Acquire GIL for Python operations
        PyGILState_STATE gstate = PyGILState_Ensure();

        try {
            // 1. Prepare Inputs
            // Calculate max sequence length for padding
            std::vector<std::vector<int>> batch_tokens;
            batch_tokens.reserve(states.size());
            size_t max_seq_len = 0;

            for (auto* state : states) {
                // Encode state from active player's perspective
                auto tokens = encoders::TokenConverter::encode_state(*state, state->active_player_id);
                if (tokens.size() > max_seq_len) {
                    max_seq_len = tokens.size();
                }
                batch_tokens.push_back(std::move(tokens));
            }

            // Create Python lists
            PyObject* input_list = PyList_New(states.size());
            PyObject* phase_ids_list = PyList_New(states.size());

            if (!input_list || !phase_ids_list) {
                Py_XDECREF(input_list);
                Py_XDECREF(phase_ids_list);
                std::cerr << "Failed to allocate Python lists" << std::endl;
                PyGILState_Release(gstate);
                return;
            }

            for (size_t i = 0; i < states.size(); ++i) {
                const auto& tokens = batch_tokens[i];
                PyObject* token_row = PyList_New(max_seq_len);
                for (size_t j = 0; j < max_seq_len; ++j) {
                    int token = (j < tokens.size()) ? tokens[j] : encoders::TokenConverter::TOKEN_PAD;
                    PyList_SET_ITEM(token_row, j, PyLong_FromLong(token));
                }
                PyList_SET_ITEM(input_list, i, token_row); // Steals ref

                int phase_id = static_cast<int>(states[i]->current_phase);
                PyList_SET_ITEM(phase_ids_list, i, PyLong_FromLong(phase_id)); // Steals ref
            }

            // 2. Call Model
            // model(x=input_list, phase_ids=phase_ids_list)
            PyObject* args = PyTuple_New(1);
            PyTuple_SET_ITEM(args, 0, input_list); // Steals ref to input_list

            PyObject* kwargs = PyDict_New();
            // Note: phase_ids_list reference is stolen by SetItemString? No, SetItemString increments ref.
            // So we must decref phase_ids_list after adding it.
            PyDict_SetItemString(kwargs, "phase_ids", phase_ids_list);
            Py_DECREF(phase_ids_list);

            // Pass None for padding_mask if implicit (we padded with 0).
            // If the model strictly requires padding_mask, we might see performance issues or errors.
            // DuelTransformer forward: (x, padding_mask=None, phase_ids=None)
            // So we are relying on defaults for padding_mask.

            PyObject* result = PyObject_Call(model, args, kwargs);
            Py_DECREF(args);
            Py_DECREF(kwargs);

            if (!result) {
                if (PyErr_Occurred()) PyErr_Print();
                std::cerr << "Model call failed" << std::endl;
                PyGILState_Release(gstate);
                return;
            }

            // 3. Process Output
            // Expected: Tuple[logits, value]
            if (PyTuple_Check(result) && PyTuple_Size(result) == 2) {
                PyObject* logits_tensor = PyTuple_GetItem(result, 0); // Borrowed
                PyObject* value_tensor = PyTuple_GetItem(result, 1); // Borrowed

                // Convert tensors to lists (tolist())
                PyObject* logits_list = PyObject_CallMethod(logits_tensor, "tolist", nullptr);
                PyObject* value_list = PyObject_CallMethod(value_tensor, "tolist", nullptr);

                if (logits_list && value_list && PyList_Check(logits_list) && PyList_Check(value_list)) {
                    size_t batch_size = PyList_Size(logits_list);
                    if (batch_size != states.size()) {
                        std::cerr << "Batch size mismatch: expected " << states.size() << ", got " << batch_size << std::endl;
                    } else {
                        for (size_t i = 0; i < batch_size; ++i) {
                            PyObject* row = PyList_GetItem(logits_list, i); // Borrowed
                            PyObject* val_item = PyList_GetItem(value_list, i); // Borrowed

                            std::vector<float> policy;
                            if (PyList_Check(row)) {
                                size_t dim = PyList_Size(row);
                                policy.reserve(dim);
                                for (size_t k = 0; k < dim; ++k) {
                                    policy.push_back(static_cast<float>(PyFloat_AsDouble(PyList_GetItem(row, k))));
                                }
                            }

                            float value = 0.0f;
                            // Value output might be [v] (batch, 1) or just v (batch).
                            if (PyList_Check(val_item)) {
                                if (PyList_Size(val_item) > 0)
                                    value = static_cast<float>(PyFloat_AsDouble(PyList_GetItem(val_item, 0)));
                            } else {
                                value = static_cast<float>(PyFloat_AsDouble(val_item));
                            }

                            callbacks[i](policy, value);
                        }
                    }
                } else {
                     std::cerr << "Failed to convert tensors to list" << std::endl;
                     if (PyErr_Occurred()) PyErr_Print();
                }
                Py_XDECREF(logits_list);
                Py_XDECREF(value_list);
            } else {
                std::cerr << "Model output is not a 2-element tuple" << std::endl;
            }

            Py_DECREF(result);

        } catch (const std::exception& e) {
            std::cerr << "Exception in BatchEvaluator: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in BatchEvaluator" << std::endl;
        }

        PyGILState_Release(gstate);
    }

}
