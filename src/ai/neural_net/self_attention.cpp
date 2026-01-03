#include "self_attention.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>

namespace dm::ai::neural_net {

    SelfAttention::SelfAttention(int embed_dim, int num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads),
          W_q_(embed_dim, embed_dim), W_k_(embed_dim, embed_dim),
          W_v_(embed_dim, embed_dim), W_o_(embed_dim, embed_dim) {

        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }
        head_dim_ = embed_dim / num_heads;
        initialize_weights();
    }

    void SelfAttention::initialize_weights() {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim_));

        for (auto& v : W_q_.data) v = dist(gen);
        for (auto& v : W_k_.data) v = dist(gen);
        for (auto& v : W_v_.data) v = dist(gen);
        for (auto& v : W_o_.data) v = dist(gen);
    }

    Tensor2D SelfAttention::matmul(const Tensor2D& A, const Tensor2D& B) {
        if (A.cols != B.rows) {
            throw std::invalid_argument("Matrix dimension mismatch in matmul");
        }
        Tensor2D C(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; ++k) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }
        return C;
    }

    float SelfAttention::elu_plus_one(float x) const {
        // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        // PyTorch default alpha = 1.0
        float elu = (x > 0) ? x : (std::exp(x) - 1.0f);
        return elu + 1.0f;
    }

    Tensor2D SelfAttention::forward(const Tensor2D& input, const std::vector<bool>& mask) {
        int seq_len = input.rows;
        if (input.cols != embed_dim_) {
             throw std::invalid_argument("Input embedding dimension mismatch");
        }

        // 1. Linear projections
        Tensor2D Q_proj = matmul(input, W_q_);
        Tensor2D K_proj = matmul(input, W_k_);
        Tensor2D V_proj = matmul(input, W_v_);

        Tensor2D output(seq_len, embed_dim_);

        // Multi-head Linear Attention processing
        // We iterate heads and then compute the linear attention formula:
        // Out_i = (Sum_j (Q_i * K_j * V_j)) / (Sum_j (Q_i * K_j))
        // where Q, K are mapped by phi(x) = elu(x) + 1

        for (int h = 0; h < num_heads_; ++h) {
            // Extract and map Q, K, V for this head
            // Dimensions: [seq_len, head_dim]

            // We can accumulate KV and K sums directly to avoid storing full matrices if memory constrained,
            // but for clarity we'll compute transient values.

            // KV accumulator: [head_dim, head_dim] -> sum(K^T * V)
            std::vector<float> KV_sum(head_dim_ * head_dim_, 0.0f);

            // K sum accumulator: [head_dim] -> sum(K)
            std::vector<float> K_sum(head_dim_, 0.0f);

            // Temporary Q buffer for this head: [seq_len, head_dim]
            std::vector<float> Q_head(seq_len * head_dim_);

            // First pass: Compute K, V, update accumulators
            for (int j = 0; j < seq_len; ++j) {
                bool is_valid = (j < (int)mask.size()) ? mask[j] : true;
                if (!is_valid) continue; // Skip masked tokens

                // Pointers to K, V rows for this head at step j
                // K_proj row j, start index h*head_dim
                const float* k_ptr = &K_proj.data[j * embed_dim_ + h * head_dim_];
                const float* v_ptr = &V_proj.data[j * embed_dim_ + h * head_dim_];

                // Map K values: phi(K) = elu(K) + 1
                std::vector<float> k_mapped(head_dim_);
                for(int d=0; d<head_dim_; ++d) {
                    k_mapped[d] = elu_plus_one(k_ptr[d]);
                    K_sum[d] += k_mapped[d];
                }

                // Update KV_sum += k_mapped^T * v
                for(int r=0; r<head_dim_; ++r) {
                    for(int c=0; c<head_dim_; ++c) {
                        KV_sum[r * head_dim_ + c] += k_mapped[r] * v_ptr[c];
                    }
                }
            }

            // Second pass: Compute Q, calculate output
            for (int i = 0; i < seq_len; ++i) {
                // Prepare Q mapped
                const float* q_ptr = &Q_proj.data[i * embed_dim_ + h * head_dim_];
                std::vector<float> q_mapped(head_dim_);
                for(int d=0; d<head_dim_; ++d) {
                    q_mapped[d] = elu_plus_one(q_ptr[d]);
                }

                // Numerator: Q * KV_sum -> [1, head_dim] * [head_dim, head_dim] -> [1, head_dim]
                std::vector<float> num(head_dim_, 0.0f);
                for(int d=0; d<head_dim_; ++d) { // output dim (cols of KV)
                    float sum = 0.0f;
                    for(int k=0; k<head_dim_; ++k) { // reduction dim (rows of KV, cols of Q)
                         sum += q_mapped[k] * KV_sum[k * head_dim_ + d];
                    }
                    num[d] = sum;
                }

                // Denominator: Q * K_sum -> [1, head_dim] * [head_dim, 1] -> scalar
                float den = 0.0f;
                for(int d=0; d<head_dim_; ++d) {
                    den += q_mapped[d] * K_sum[d];
                }
                den += 1e-6f; // Epsilon

                // Result for this head: num / den
                for(int d=0; d<head_dim_; ++d) {
                    output.at(i, h * head_dim_ + d) = num[d] / den;
                }
            }
        }

        // Final linear projection
        return matmul(output, W_o_);
    }

} // namespace dm::ai::neural_net
