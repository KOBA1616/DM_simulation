#ifndef DM_AI_NEURAL_NET_SELF_ATTENTION_HPP
#define DM_AI_NEURAL_NET_SELF_ATTENTION_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>

namespace dm::ai::neural_net {

    // Simple tensor structure for standard C++ implementation
    struct Tensor2D {
        std::vector<float> data;
        int rows;
        int cols;

        Tensor2D(int r, int c, float val = 0.0f) : rows(r), cols(c) {
            data.resize(r * c, val);
        }

        float& at(int r, int c) { return data[r * cols + c]; }
        const float& at(int r, int c) const { return data[r * cols + c]; }
    };

    /**
     * Implements Linear Attention (O(N) complexity) to match Python implementation.
     * Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
     */
    class SelfAttention {
    public:
        SelfAttention(int embed_dim, int num_heads);

        // Forward pass: Input [SeqLen, EmbedDim] -> Output [SeqLen, EmbedDim]
        Tensor2D forward(const Tensor2D& input, const std::vector<bool>& mask);

        // Parameter initialization (random for now, or loadable)
        void initialize_weights();

    private:
        int embed_dim_;
        int num_heads_;
        int head_dim_;

        // Weights
        Tensor2D W_q_;
        Tensor2D W_k_;
        Tensor2D W_v_;
        Tensor2D W_o_;

        // Helper for matrix multiplication
        Tensor2D matmul(const Tensor2D& A, const Tensor2D& B);

        // ELU + 1 activation function for feature map
        float elu_plus_one(float x) const;
    };

} // namespace dm::ai::neural_net

#endif // DM_AI_NEURAL_NET_SELF_ATTENTION_HPP
