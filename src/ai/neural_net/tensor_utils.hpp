#ifndef DM_AI_NEURAL_NET_TENSOR_UTILS_HPP
#define DM_AI_NEURAL_NET_TENSOR_UTILS_HPP

#include "types.hpp"

namespace dm::ai::neural_net {

    // A collection of basic operations for inference
    // Not optimized for performance (no BLAS), just correctness for Phase 4 proof-of-concept
    class Ops {
    public:
        // C = A * B
        // Assumes A [M, K], B [K, N] -> C [M, N]
        static Tensor matmul(const Tensor& A, const Tensor& B) {
            if (A.dim() != 2 || B.dim() != 2 || A.shape[1] != B.shape[0]) {
                throw std::invalid_argument("Matmul shape mismatch");
            }
            int M = A.shape[0];
            int K = A.shape[1];
            int N = B.shape[1];

            Tensor C({M, N});

            // Naive O(M*N*K)
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A.data[m * K + k] * B.data[k * N + n];
                    }
                    C.data[m * N + n] = sum;
                }
            }
            return C;
        }

        static void add_inplace(Tensor& A, const Tensor& B) {
            if (A.data.size() != B.data.size()) throw std::invalid_argument("Add size mismatch");
            for (size_t i = 0; i < A.data.size(); ++i) A.data[i] += B.data[i];
        }

        static void relu_inplace(Tensor& A) {
            for (float& v : A.data) if (v < 0) v = 0;
        }

        static void elu_inplace(Tensor& A) {
            for (float& v : A.data) if (v < 0) v = std::exp(v) - 1.0f;
        }

        static void gelu_inplace(Tensor& A) {
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float c1 = 0.7978845608f; // sqrt(2/pi)
            const float c2 = 0.044715f;
            for (float& x : A.data) {
                float cube = x * x * x;
                float inner = c1 * (x + c2 * cube);
                float th = std::tanh(inner);
                x = 0.5f * x * (1.0f + th);
            }
        }

        static void tanh_inplace(Tensor& A) {
             for (float& v : A.data) v = std::tanh(v);
        }

        // LayerNorm: (x - mean) / std * gamma + beta
        static void layer_norm(Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) {
            // Assumes x is [Batch, Dim] or [Seq, Dim] and gamma/beta are [Dim]
            int dim = x.shape.back();
            int batches = x.size() / dim;

            for (int b = 0; b < batches; ++b) {
                float mean = 0.0f;
                float var = 0.0f;
                int offset = b * dim;

                for (int i = 0; i < dim; ++i) mean += x.data[offset + i];
                mean /= dim;

                for (int i = 0; i < dim; ++i) {
                    float d = x.data[offset + i] - mean;
                    var += d * d;
                }
                var /= dim;
                float std_inv = 1.0f / std::sqrt(var + eps);

                for (int i = 0; i < dim; ++i) {
                    float norm = (x.data[offset + i] - mean) * std_inv;
                    x.data[offset + i] = norm * gamma.data[i] + beta.data[i];
                }
            }
        }

        static Tensor embedding_lookup(const std::vector<int>& indices, const Tensor& weights) {
            // weights: [Vocab, Dim]
            // indices: [Seq]
            // Output: [Seq, Dim]
            int seq_len = indices.size();
            int dim = weights.shape[1];
            Tensor out({seq_len, dim});

            for(int i=0; i<seq_len; ++i) {
                int idx = indices[i];
                if (idx < 0 || idx >= weights.shape[0]) idx = 0; // Boundary check
                for(int d=0; d<dim; ++d) {
                    out.data[i*dim + d] = weights.data[idx*dim + d];
                }
            }
            return out;
        }
    };

} // namespace dm::ai::neural_net

#endif // DM_AI_NEURAL_NET_TENSOR_UTILS_HPP
