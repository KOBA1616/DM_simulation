#ifndef DM_AI_NEURAL_NET_TRANSFORMER_HPP
#define DM_AI_NEURAL_NET_TRANSFORMER_HPP

#include "types.hpp"
#include "tensor_utils.hpp"
#include <map>
#include <string>

namespace dm::ai::neural_net {

    // Simple Linear Layer
    class Linear {
    public:
        Tensor weight; // [Out, In] (Transposed compared to matmul input A usually, Pytorch stores [Out, In])
        Tensor bias;
        bool has_bias;

        Linear(int in_features, int out_features, bool bias_ = true);
        Tensor forward(const Tensor& input); // Input [N, In] -> [N, Out]
    };

    class LayerNorm {
    public:
        Tensor weight; // Gamma
        Tensor bias;   // Beta
        LayerNorm(int dim);
        void forward_inplace(Tensor& input);
    };

    class LinearAttention {
    public:
        int dim;
        int heads;
        int dim_head;
        int inner_dim;

        Linear to_qkv;
        Linear to_out_l; // First part of to_out Sequential
        // No dropout in inference

        LinearAttention(int dim, int heads, int dim_head);
        Tensor forward(const Tensor& x, const std::vector<float>& mask); // x: [Seq, Dim], mask: [Seq] (1.0 or 0.0)
    };

    class TransformerBlock {
    public:
        LayerNorm norm1;
        LinearAttention attn;
        LayerNorm norm2;

        // FFN
        Linear ffn_1;
        Linear ffn_2;

        TransformerBlock(int dim, int heads, int dim_head, int ffn_mult = 4);
        void forward_inplace(Tensor& x, const std::vector<float>& mask);
    };

    class TransformerModel {
    public:
        TransformerModel(int embedding_dim, int depth, int heads, int vocab_size, int max_seq_len, int action_space);

        // Load weights from a flat float vector (e.g. from binary dump)
        // Returns true if size matched
        bool load_weights(const std::vector<float>& all_weights);

        // Inference
        // tokens: [Seq]
        // mask: [Seq] (1 for valid, 0 for pad)
        // Returns {policy_logits, value}
        std::pair<std::vector<float>, float> forward(const std::vector<int>& tokens, const std::vector<int>& mask);

    private:
        int embedding_dim_;
        int max_seq_len_;

        Tensor card_embedding_; // [Vocab, Dim]
        Tensor pos_embedding_;  // [1, MaxSeq, Dim] -> we treat as [MaxSeq, Dim]

        std::vector<TransformerBlock> layers_;
        LayerNorm norm_;

        Linear policy_head_;

        // Value head sequential: Linear -> ReLU -> Linear -> Tanh
        Linear value_head_1_;
        Linear value_head_2_;
    };

} // namespace dm::ai::neural_net

#endif // DM_AI_NEURAL_NET_TRANSFORMER_HPP
