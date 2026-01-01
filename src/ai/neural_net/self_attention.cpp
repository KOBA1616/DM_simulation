#include "self_attention.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

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

    Tensor2D SelfAttention::matmul_transpose_b(const Tensor2D& A, const Tensor2D& B) {
        if (A.cols != B.cols) {
            throw std::invalid_argument("Matrix dimension mismatch in matmul_transpose_b");
        }
        Tensor2D C(A.rows, B.rows);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.rows; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; ++k) {
                    sum += A.at(i, k) * B.at(j, k);
                }
                C.at(i, j) = sum;
            }
        }
        return C;
    }

    void SelfAttention::softmax_row(std::vector<float>& row) {
        float max_val = -1e9;
        for (float v : row) if (v > max_val) max_val = v;

        float sum = 0.0f;
        for (float& v : row) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (float& v : row) v /= sum;
    }

    Tensor2D SelfAttention::forward(const Tensor2D& input, const std::vector<bool>& mask) {
        int seq_len = input.rows;
        if (input.cols != embed_dim_) {
             throw std::invalid_argument("Input embedding dimension mismatch");
        }

        // 1. Linear projections
        Tensor2D Q = matmul(input, W_q_);
        Tensor2D K = matmul(input, W_k_);
        Tensor2D V = matmul(input, W_v_);

        Tensor2D output(seq_len, embed_dim_);

        // Multi-head attention
        for (int h = 0; h < num_heads_; ++h) {
            // Extract Q, K, V for this head
            // In a real optimized implementation, this is done by reshaping/striding
            // Here we do it explicitly for clarity/simplicity in this stub

            // Scaled Dot-Product Attention
            // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

            // Compute scores = QK^T
            // Q_h: [seq_len, head_dim], K_h: [seq_len, head_dim]
            // scores: [seq_len, seq_len]

            for (int i = 0; i < seq_len; ++i) {
                // Check mask for query position (if needed, usually we attend FROM valid positions)
                // But typically masking is done on keys (to ignore padding)

                std::vector<float> attention_scores(seq_len);

                for (int j = 0; j < seq_len; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) {
                         // Indexing: row i, col (h * head_dim + d)
                         float q_val = Q.at(i, h * head_dim_ + d);
                         float k_val = K.at(j, h * head_dim_ + d);
                         dot += q_val * k_val;
                    }
                    attention_scores[j] = dot / std::sqrt((float)head_dim_);

                    // Apply mask (mask[j] == false means padding/invalid)
                    if (j < (int)mask.size() && !mask[j]) {
                        attention_scores[j] = -1e9;
                    }
                }

                softmax_row(attention_scores);

                // Weighted sum of V
                for (int d = 0; d < head_dim_; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        val += attention_scores[j] * V.at(j, h * head_dim_ + d);
                    }
                    // Write to output (concatenated implicitly)
                    output.at(i, h * head_dim_ + d) = val;
                }
            }
        }

        // Final linear projection
        return matmul(output, W_o_);
    }

} // namespace dm::ai::neural_net
