#include "transformer.hpp"
#include <iostream>

namespace dm::ai::neural_net {

    // --- Linear ---
    Linear::Linear(int in_features, int out_features, bool bias_)
        : has_bias(bias_) {
        weight = Tensor({in_features, out_features}); // We store as [In, Out] for easier matmul A*W
        if (has_bias) bias = Tensor({out_features});
    }

    Tensor Linear::forward(const Tensor& input) {
        // Input: [N, In]
        // Weight: [In, Out]
        // Output: [N, Out]
        Tensor out = Ops::matmul(input, weight);

        if (has_bias) {
            int N = out.shape[0];
            int Out = out.shape[1];
            for(int i=0; i<N; ++i) {
                for(int j=0; j<Out; ++j) {
                    out.data[i*Out + j] += bias.data[j];
                }
            }
        }
        return out;
    }

    // --- LayerNorm ---
    LayerNorm::LayerNorm(int dim) {
        weight = Tensor({dim}, 1.0f);
        bias = Tensor({dim}, 0.0f);
    }

    void LayerNorm::forward_inplace(Tensor& input) {
        Ops::layer_norm(input, weight, bias);
    }

    // --- LinearAttention ---
    LinearAttention::LinearAttention(int dim, int heads, int dim_head)
        : dim(dim), heads(heads), dim_head(dim_head), inner_dim(heads * dim_head),
          to_qkv(dim, inner_dim * 3, false),
          to_out_l(inner_dim, dim, true) // PyTorch Linear has bias by default if not specified, network_v2 doesn't specify for to_out, so it has bias.
    {
    }

    Tensor LinearAttention::forward(const Tensor& x, const std::vector<float>& mask) {
        // x: [Seq, Dim]
        int seq_len = x.shape[0];

        // 1. QKV
        Tensor qkv = to_qkv.forward(x); // [Seq, InnerDim*3]

        // Split Q, K, V and reshape/transpose for heads
        // Logic:
        // qkv: [Seq, 3*H*D]
        // q: [Seq, H, D], k: [Seq, H, D], v: [Seq, H, D]
        // Note: PyTorch LinearAttention uses ELU+1

        // We will process head by head to avoid large 4D tensors in naive C++
        // Output accumulator
        Tensor out_all_heads({seq_len, inner_dim}); // [Seq, H*D] which is [Seq, InnerDim]

        for (int h = 0; h < heads; ++h) {
            // Extract q, k, v for this head
            // qkv index map: [seq, 3, heads, dim_head] (roughly)
            // But chunk(3) splits the last dim into 3 parts: [0..ID), [ID..2ID), [2ID..3ID)
            // Each part is [Seq, InnerDim]. Reshape to [Seq, Heads, DimHead]

            // Temporary buffers for this head: [Seq, DimHead]
            Tensor Q({seq_len, dim_head});
            Tensor K({seq_len, dim_head});
            Tensor V({seq_len, dim_head});

            for (int s = 0; s < seq_len; ++s) {
                int base_offset = s * (inner_dim * 3);

                // Q
                int q_start = base_offset + h * dim_head;
                for(int d=0; d<dim_head; ++d) Q.data[s*dim_head + d] = qkv.data[q_start + d];

                // K
                int k_start = base_offset + inner_dim + h * dim_head;
                for(int d=0; d<dim_head; ++d) K.data[s*dim_head + d] = qkv.data[k_start + d];

                // V
                int v_start = base_offset + 2*inner_dim + h * dim_head;
                for(int d=0; d<dim_head; ++d) V.data[s*dim_head + d] = qkv.data[v_start + d];
            }

            // ELU + 1 on Q and K
            Ops::elu_inplace(Q);
            Ops::elu_inplace(K);
            for(float& v : Q.data) v += 1.0f;
            for(float& v : K.data) v += 1.0f;

            // Apply Mask to K, V
            if (!mask.empty()) {
                for (int s = 0; s < seq_len; ++s) {
                    if (mask[s] == 0.0f) {
                        for(int d=0; d<dim_head; ++d) {
                            K.data[s*dim_head + d] = 0.0f;
                            V.data[s*dim_head + d] = 0.0f;
                        }
                    }
                }
            }

            // Linear Attention Core:
            // KV = einsum(K, V) -> K^T * V -> [DimHead, DimHead]
            // Tensor K is [Seq, D]. We treat as [Seq, D].
            // We want [D, D] output.
            // Transpose K manually or optimize
            // K^T is [D, Seq].
            // KV = K^T * V

            // Compute KV [D, D]
            std::vector<float> KV(dim_head * dim_head, 0.0f);
            for (int d1 = 0; d1 < dim_head; ++d1) {
                for (int d2 = 0; d2 < dim_head; ++d2) {
                    float sum = 0.0f;
                    for (int s = 0; s < seq_len; ++s) {
                        sum += K.data[s*dim_head + d1] * V.data[s*dim_head + d2];
                    }
                    KV[d1 * dim_head + d2] = sum;
                }
            }

            // Compute Z denominator = Q * K.sum(dim=0) (since we are Seq, D)
            // K_sum: [D]
            std::vector<float> K_sum(dim_head, 0.0f);
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < dim_head; ++d) {
                    K_sum[d] += K.data[s*dim_head + d];
                }
            }

            // Z: [Seq]
            std::vector<float> Z(seq_len);
            for(int s=0; s<seq_len; ++s) {
                float dot = 0.0f;
                for(int d=0; d<dim_head; ++d) dot += Q.data[s*dim_head + d] * K_sum[d];
                Z[s] = 1.0f / (dot + 1e-6f);
            }

            // Compute Output Numerator: Q * KV -> [Seq, D] * [D, D] -> [Seq, D]
            // We can reuse Q buffer for output numerator
            std::vector<float> Num(seq_len * dim_head);
            for(int s=0; s<seq_len; ++s) {
                for(int d_out=0; d_out<dim_head; ++d_out) {
                    float sum = 0.0f;
                    for(int d_in=0; d_in<dim_head; ++d_in) {
                        sum += Q.data[s*dim_head + d_in] * KV[d_in * dim_head + d_out];
                    }
                    Num[s*dim_head + d_out] = sum;
                }
            }

            // Final Output for head = Num * Z
            for(int s=0; s<seq_len; ++s) {
                for(int d=0; d<dim_head; ++d) {
                    out_all_heads.data[s * inner_dim + h * dim_head + d] = Num[s*dim_head + d] * Z[s];
                }
            }
        }

        // Final Projection
        return to_out_l.forward(out_all_heads);
    }

    // --- TransformerBlock ---
    TransformerBlock::TransformerBlock(int dim, int heads, int dim_head, int ffn_mult)
        : norm1(dim), attn(dim, heads, dim_head), norm2(dim),
          ffn_1(dim, dim * ffn_mult, true),
          ffn_2(dim * ffn_mult, dim, true)
    {
    }

    void TransformerBlock::forward_inplace(Tensor& x, const std::vector<float>& mask) {
        // x: [Seq, Dim]

        // 1. Attention Branch
        Tensor x_norm = x; // Copy
        norm1.forward_inplace(x_norm);
        Tensor attn_out = attn.forward(x_norm, mask);
        Ops::add_inplace(x, attn_out); // Residual

        // 2. FFN Branch
        x_norm = x; // Copy
        norm2.forward_inplace(x_norm);

        Tensor ffn_out = ffn_1.forward(x_norm);
        Ops::gelu_inplace(ffn_out);
        ffn_out = ffn_2.forward(ffn_out);

        Ops::add_inplace(x, ffn_out); // Residual
    }

    // --- TransformerModel ---
    TransformerModel::TransformerModel(int embedding_dim, int depth, int heads, int vocab_size, int max_seq_len, int action_space)
        : embedding_dim_(embedding_dim), max_seq_len_(max_seq_len),
          norm_(embedding_dim),
          policy_head_(embedding_dim, action_space, true),
          value_head_1_(embedding_dim, 128, true),
          value_head_2_(128, 1, true)
    {
        // Embeddings
        card_embedding_ = Tensor({vocab_size, embedding_dim});
        pos_embedding_ = Tensor({max_seq_len, embedding_dim});

        int dim_head = embedding_dim / heads;
        for(int i=0; i<depth; ++i) {
            layers_.emplace_back(embedding_dim, heads, dim_head);
        }
    }

    bool TransformerModel::load_weights(const std::vector<float>& all_weights) {
        size_t offset = 0;
        auto load_tensor = [&](Tensor& t) {
            if (offset + t.data.size() > all_weights.size()) return false;
            std::copy(all_weights.begin() + offset, all_weights.begin() + offset + t.data.size(), t.data.begin());
            offset += t.data.size();
            return true;
        };

        // Order must match export script!
        if (!load_tensor(card_embedding_)) return false;
        if (!load_tensor(pos_embedding_)) return false;

        for(auto& layer : layers_) {
            // Norm1
            if (!load_tensor(layer.norm1.weight)) return false;
            if (!load_tensor(layer.norm1.bias)) return false;
            // Attn QKV
            if (!load_tensor(layer.attn.to_qkv.weight)) return false;
            // Linear in pytorch is [Out, In], but we stored as [In, Out] for matmul
            // We assume export script handles transpose if needed.
            // Actually, my Linear impl expects [In, Out]. PyTorch stores [Out, In].
            // If export script flattens PyTorch weight, it is row-major [Out, In].
            // I need to transpose it during load or ensure export transposes.
            // Let's assume export script exports in order (Row 0, Row 1...) which corresponds to [Out, In].
            // My Tensor is flattened. A.matmul(W). A=[N, In], W=[In, Out].
            // If I load [Out, In] directly into W data, it will be interpreted as [In, Out] (col-major) or row-major [In, Out]?
            // Tensor logic: data[row * cols + col].
            // If I want W[In, Out], data should be row 0 (size Out), row 1 (size Out)...
            // PyTorch W is [Out, In]. Row 0 is weights for output neuron 0.
            // So PyTorch data is: (w_0,0 ... w_0,In), (w_1,0 ... w_1,In).
            // This is effectively W^T in math terms if W is on right. xW.
            // x[1, In] * W[In, Out].
            // So I need the transpose of PyTorch weights.
            // I will assume the export script does NOT transpose, so I must transpose here.
            // Wait, for simplicity, I will implement a `load_linear_weight` helper.

            // Re-think:
            // PyTorch: y = x A^T + b. A is [Out, In].
            // My C++: y = x W + b. W is [In, Out].
            // So W should be A^T.
            // If I load A (flat) = [Row0(In), Row1(In)...], that is [Out, In].
            // If I just copy to my W (flat), my W is treated as [In, Out].
            // So W(0, 1) becomes A(0, 1).
            // My W(r, c) is index r*Out + c.
            // Loaded A(r, c) is index r*In + c.
            // They don't map directly if In != Out.
            // So I definitely need to transpose.

            // However, implementing transpose here is tedious.
            // I will mandate the Python export script to transpose the weights for Linear layers.

            // Attn Out
            if (!load_tensor(layer.attn.to_out_l.weight)) return false;
            if (!load_tensor(layer.attn.to_out_l.bias)) return false;
            // Norm2
            if (!load_tensor(layer.norm2.weight)) return false;
            if (!load_tensor(layer.norm2.bias)) return false;
            // FFN1
            if (!load_tensor(layer.ffn_1.weight)) return false;
            if (!load_tensor(layer.ffn_1.bias)) return false;
            // FFN2
            if (!load_tensor(layer.ffn_2.weight)) return false;
            if (!load_tensor(layer.ffn_2.bias)) return false;
        }

        if (!load_tensor(norm_.weight)) return false;
        if (!load_tensor(norm_.bias)) return false;

        if (!load_tensor(policy_head_.weight)) return false;
        if (!load_tensor(policy_head_.bias)) return false;

        if (!load_tensor(value_head_1_.weight)) return false;
        if (!load_tensor(value_head_1_.bias)) return false;
        if (!load_tensor(value_head_2_.weight)) return false;
        if (!load_tensor(value_head_2_.bias)) return false;

        return offset == all_weights.size();
    }

    std::pair<std::vector<float>, float> TransformerModel::forward(const std::vector<int>& tokens, const std::vector<int>& mask) {
        // tokens: [Seq]
        int seq_len = tokens.size();
        if (seq_len > max_seq_len_) seq_len = max_seq_len_;

        std::vector<int> t_sub(tokens.begin(), tokens.begin() + seq_len);

        // 1. Embedding
        Tensor x = Ops::embedding_lookup(t_sub, card_embedding_); // [Seq, Dim]

        // Add Positional (slice)
        for(int s=0; s<seq_len; ++s) {
            for(int d=0; d<embedding_dim_; ++d) {
                x.data[s*embedding_dim_ + d] += pos_embedding_.data[s*embedding_dim_ + d];
            }
        }

        // Float Mask
        std::vector<float> f_mask(seq_len);
        for(int i=0; i<seq_len; ++i) f_mask[i] = (i < (int)mask.size()) ? (float)mask[i] : 0.0f;

        // 2. Layers
        for(auto& layer : layers_) {
            layer.forward_inplace(x, f_mask);
        }

        // 3. Norm
        norm_.forward_inplace(x);

        // 4. Pooling (Mean with mask)
        Tensor pooled({1, embedding_dim_}, 0.0f);
        float mask_sum = 0.0f;
        for(int s=0; s<seq_len; ++s) {
            if (f_mask[s] > 0.5f) {
                mask_sum += 1.0f;
                for(int d=0; d<embedding_dim_; ++d) {
                    pooled.data[d] += x.data[s*embedding_dim_ + d];
                }
            }
        }
        if (mask_sum < 1.0f) mask_sum = 1.0f;
        for(float& v : pooled.data) v /= mask_sum;

        // 5. Heads
        Tensor p_logits = policy_head_.forward(pooled);

        Tensor v_h1 = value_head_1_.forward(pooled);
        Ops::relu_inplace(v_h1);
        Tensor v_out = value_head_2_.forward(v_h1);
        Ops::tanh_inplace(v_out);

        // Softmax policy
        // (Simple Softmax)
        float max_l = -1e9;
        for(float v : p_logits.data) if(v > max_l) max_l = v;
        float sum_exp = 0.0f;
        for(float& v : p_logits.data) {
            v = std::exp(v - max_l);
            sum_exp += v;
        }
        for(float& v : p_logits.data) v /= sum_exp;

        return {p_logits.data, v_out.data[0]};
    }

} // namespace dm::ai::neural_net
