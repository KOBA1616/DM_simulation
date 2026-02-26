#ifndef DM_AI_NEURAL_NET_TYPES_HPP
#define DM_AI_NEURAL_NET_TYPES_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace dm::ai::neural_net {

    // Simple Tensor Structure
    // For Transformer:
    // 2D: [Batch, Dim] or [Seq, Dim]
    // 3D: [Batch, Seq, Dim] (Often flattened to 2D for linear layers)
    struct Tensor {
        std::vector<float> data;
        std::vector<int> shape;

        Tensor() = default;
        Tensor(const std::vector<int>& s, float val = 0.0f) : shape(s) {
            int size = 1;
            for (int d : s) size *= d;
            data.resize(size, val);
        }

        int size() const { return data.size(); }
        int dim() const { return shape.size(); }

        float& at(int i) { return data[i]; }
        const float& at(int i) const { return data[i]; }

        float& at(const std::vector<int>& idx) {
            int offset = 0;
            int stride = 1;
            for (int i = dim() - 1; i >= 0; --i) {
                offset += idx[i] * stride;
                stride *= shape[i];
            }
            return data[offset];
        }
    };

} // namespace dm::ai::neural_net

#endif // DM_AI_NEURAL_NET_TYPES_HPP
