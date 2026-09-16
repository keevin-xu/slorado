#include "TxModel.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/options/padding.h>
#include <torch/types.h>
#include <torch/version.h>

#include <cmath>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <stdexcept>
#include <string>

using namespace torch::nn;
using Slice = torch::indexing::Slice;

#ifdef OPENFISH_DUMP
// Debug-only tensor dump for validating NPU kernels against this CPU path
// (openfish/npu). Build with -DOPENFISH_DUMP; runtime-enabled by OPENFISH_DUMP_DIR.
// Dumps encoder layers [0, OPENFISH_DUMP_LAYERS) (default 1) of the first forward
// pass as .npy, keeping the leading OPENFISH_DUMP_ROWS batch rows (default 4) of
// activations, plus <dir>/manifest.tsv with full shapes. OPENFISH_DUMP_EXIT=1 exits
// the process after the last dumped layer. Assumes one runner (-r 1, the default).
#include <cstdio>
#include <cstdlib>

static int dump_layer = -1;
static int64_t dump_encoder_calls = 0;

static int64_t dump_env_int(const char *name, int64_t dflt) {
    const char *v = std::getenv(name);
    return v ? std::atoll(v) : dflt;
}

static std::string dump_shape_str(const torch::Tensor &t, const char *sep) {
    std::string s;
    for (int64_t i = 0; i < t.dim(); ++i) {
        s += (i ? sep : "") + std::to_string(t.size(i));
    }
    return s;
}

static std::string dump_stride_str(const torch::Tensor &t) {
    std::string s;
    for (int64_t i = 0; i < t.dim(); ++i) {
        s += (i ? "," : "") + std::to_string(t.stride(i));
    }
    return s;
}

// batched: activation whose dim 0 is the batch (sliced to the leading rows);
// otherwise a parameter/buffer saved whole.
static void dump_tensor(const char *site, const torch::Tensor &t, bool batched = true) {
    const char *dir = std::getenv("OPENFISH_DUMP_DIR");
    if (!dir || dump_layer < 0 || !t.defined()) {
        return;
    }
    torch::Tensor s = t.detach();
    if (batched && s.dim() > 0) {
        s = s.narrow(0, 0, std::min<int64_t>(s.size(0), dump_env_int("OPENFISH_DUMP_ROWS", 4)));
    }
    s = s.to(torch::kCPU).contiguous();

    const char *descr = nullptr;
    switch (s.scalar_type()) {
        case torch::kFloat32: descr = "<f4"; break;
        case torch::kFloat16: descr = "<f2"; break;
        case torch::kFloat64: descr = "<f8"; break;
        case torch::kInt64: descr = "<i8"; break;
        case torch::kInt32: descr = "<i4"; break;
        case torch::kBool: descr = "|b1"; break;
        default:
            ERROR("OPENFISH_DUMP: unsupported dtype for %s", site);
            return;
    }

    const std::string name = "L" + std::to_string(dump_layer) + "_" + site + ".npy";
    const std::string path = std::string(dir) + "/" + name;
    std::string shape = "(" + dump_shape_str(s, ", ") + (s.dim() == 1 ? ",)" : ")");
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    header.append(64 - (10 + header.size() + 1) % 64, ' ');
    header += '\n';

    FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        ERROR("OPENFISH_DUMP: cannot open %s", path.c_str());
        return;
    }
    const uint16_t header_len = header.size();
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, fp);
    std::fwrite(&header_len, sizeof(header_len), 1, fp);
    std::fwrite(header.data(), 1, header.size(), fp);
    std::fwrite(s.data_ptr(), s.element_size(), s.numel(), fp);
    std::fclose(fp);

    const std::string manifest = std::string(dir) + "/manifest.tsv";
    FILE *mf = std::fopen(manifest.c_str(), "a");
    if (mf) {
        std::fprintf(mf, "%d\t%s\t%s\t%s\t%s\t%s\t%s\n", dump_layer, site,
                     dump_shape_str(t, "x").c_str(), dump_shape_str(s, "x").c_str(),
                     dump_stride_str(t).c_str(), descr, name.c_str());
        std::fclose(mf);
    }
}
#endif

void apply_rounding(torch::Tensor &t, int remove_bits) {
    // Round Float16 tensor elements such that the last `remove_bits` of the mantissa are 0s.
    // TODO: this is slightly dangerous as it will turn numbers close to +/-65304 into +/-inf
    t.view(torch::kI16).add_(1 << (remove_bits - 1));
    t.view(torch::kI16).bitwise_and_(0x10000 - (1 << remove_bits));
}

torch::Tensor scaled_dot_product_attention_naive(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &mask
) {
    auto matmul_qk = torch::matmul(q, k.transpose(-2, -1));

    auto d_k = k.size(-1);
    matmul_qk = matmul_qk / std::sqrt(d_k);

    if (mask.defined()) {
        matmul_qk = matmul_qk + (mask.logical_not() * -1e9);
    }

    auto weights = torch::softmax(matmul_qk, -1);
    return torch::matmul(weights, v);
}

RMSNormImpl::RMSNormImpl(int hidden_size_) : hidden_size(hidden_size_) {
    weight = torch::ones({hidden_size});
    register_parameter("weight", weight, false);
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    torch::Tensor rstd = torch::rsqrt(x.square().mean(-1, true).add_(eps));
    x.mul_(rstd).mul_(weight);
    return x;
}

GatedMLPImpl::GatedMLPImpl(int in_features_, int hidden_features_)
    : in_features(in_features_), hidden_features(hidden_features_) {
    fc1 = register_module("fc1", Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

torch::Tensor GatedMLPImpl::forward(const torch::Tensor &x) {
    torch::Tensor t;
    t = at::linear(x, fc1->weight, fc1->bias);
#ifdef OPENFISH_DUMP
    dump_tensor("ff_in", x);
    dump_tensor("ff_fc1_weight", fc1->weight, false);
    dump_tensor("ff_fc2_weight", fc2->weight, false);
    dump_tensor("ff_fc1_out", t);
#endif
#ifdef USE_GPU
    auto M = t.size(0) * t.size(1);
    auto K = t.size(2) / 2;
    auto silu_o = torch::empty({t.size(0), t.size(1), K}, t.options());
    openfish_silu_mul_gpu(t.data_ptr(), silu_o.data_ptr(), M, K);
    t = silu_o;
#else
    const auto chunks = t.chunk(2, -1);
    const auto &y = chunks[0];
    const auto &gate = chunks[1];
    t = functional::silu(gate).mul_(y);
#endif
#ifdef OPENFISH_DUMP
    dump_tensor("ff_silu_mul_out", t);
    torch::Tensor fc2_out = at::linear(t, fc2->weight, fc2->bias);
    dump_tensor("ff_fc2_out", fc2_out);
    return fc2_out;
#else
    return at::linear(t, fc2->weight, fc2->bias);
#endif
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(
    int dim_,
    float theta_,
    int max_seq_len_,
    const torch::TensorOptions &options_,
    tx_stats_t *stats
) :
    dim(dim_),
    max_seq_len(max_seq_len_),
    theta(theta_),
    options(options_)
{
    stats_ = stats;
    auto inv_freq = torch::pow(theta, torch::arange(0, dim, 2, options) / dim).reciprocal();
    torch::Tensor freqs = torch::arange(max_seq_len, options).outer(inv_freq);

    auto cos = torch::cos(freqs).to(torch::kFloat32).contiguous();
    auto sin = torch::sin(freqs).to(torch::kFloat32).contiguous();
    cos_buf = cos;
    sin_buf = sin;
};

torch::Tensor RotaryEmbeddingImpl::forward(torch::Tensor &qkv) {
    assert_forward_dims(qkv);
    const int batch_size = qkv.size(0);
    const int seqlen = qkv.size(1);
    const int nheads = qkv.size(3);
    const int head_dim = qkv.size(4);
    const int rotary_dim = 32;
    const int stride_batch = qkv.stride(0);
    const int stride_seq = qkv.stride(1);
    const int stride_head = qkv.stride(3);

    auto qkv_chunks = qkv.chunk(3, 2);

#ifdef USE_GPU
    if (!qkv.device().is_cpu()) {
        openfish_rotary_emb_gpu(
            qkv_chunks[0].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head
        );
        
        openfish_rotary_emb_gpu(
            qkv_chunks[1].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head
        );
    } else
#endif
    {
        openfish_rotary_emb_cpu(
            qkv_chunks[0].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head,
            stats_->nthreads
        );

        openfish_rotary_emb_cpu(
            qkv_chunks[1].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head,
            stats_->nthreads
        );
    }
    
    return qkv;
}

void RotaryEmbeddingImpl::assert_forward_dims(const torch::Tensor &qkv) const {
    // Expected shape: N, seq_len, 3, nhead, head_dim
    const int64_t seq_len = qkv.size(1);
    const int64_t three = qkv.size(2);
    const int64_t head_dim = qkv.size(4);

    bool has_error = false;
    if (seq_len > max_seq_len) {
        has_error = true;
        ERROR("RotE - maximum sequence length exceeded - len:%ld max:%ld - Your chunksize may be too large", seq_len, max_seq_len);
    }
    if (three != 3) {
        has_error = true;
        ERROR("RotE - expected constant size:3 at dim:2 found:%ld", three);
    }
    if (head_dim != dim) {
        has_error = true;
        ERROR("RotE - expected head_dim size:%ld at dim:4 found:%ld", dim, head_dim);
    }
    if (has_error) {
        exit(EXIT_FAILURE);
    }
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
    int d_model_,
    int nhead_,
    bool qkv_bias_,
    bool out_bias_,
    const std::pair<int, int> &attn_window_,
    const torch::TensorOptions &options_,
    tx_stats_t *_model_stats
) :
    d_model(d_model_),
    nhead(nhead_),
    head_dim(d_model_ / nhead_),
    // TODO: this may benefit from fine-tuning. 8 gives good performance at chunk size 12k
    num_splits(12),
    attn_window(attn_window_),
    options(options_)
{
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
    const float theta = 10000.0f;
    const int64_t max_seq_len = 2048;
    rotary_emb = register_module("rotary_emb", RotaryEmbedding(head_dim, theta, max_seq_len, options, _model_stats));
    model_stats = _model_stats;
};


torch::Tensor MultiHeadAttentionImpl::get_attn_window_mask(const int64_t size) {
    const auto key = MaskKey{size, options.device()};
    if (mask_cache.find(key) == mask_cache.end()) {
        mask_cache[key] = build_attn_window_mask(size);
    }
    return mask_cache.at(key);
}

torch::Tensor MultiHeadAttentionImpl::build_attn_window_mask(const int64_t size) const {
    const auto win_upper = std::get<0>(attn_window);
    const auto win_lower = std::get<1>(attn_window);
    torch::Tensor mask = torch::ones({size, size}, options.device());
    mask.triu_(-win_upper).tril_(win_lower);
    mask = mask.to(torch::kBool);
    return mask;
};

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);

    double a, b;
    
    a = realtime();
    auto qkv = at::linear(x, wqkv->weight, wqkv->bias)
                   .view({N, T, 3, nhead, head_dim});
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_mm += b-a;
#ifdef OPENFISH_DUMP
    dump_tensor("attn_in", x);
    dump_tensor("attn_wqkv_weight", wqkv->weight, false);
    dump_tensor("attn_wqkv_bias", wqkv->bias, false);
    dump_tensor("attn_qkv_linear_out", qkv);
    dump_tensor("attn_rotary_sin", rotary_emb->sin_buf, false);
    dump_tensor("attn_rotary_cos", rotary_emb->cos_buf, false);
#endif

    a = realtime();
    qkv = rotary_emb(qkv);
#ifdef OPENFISH_DUMP
    dump_tensor("attn_qkv_rotary_out", qkv);
#endif
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_rotary_emb += b-a;

    a = realtime();
    const auto win_upper = std::get<0>(attn_window);
    const auto win_lower = std::get<1>(attn_window);

    torch::Tensor attn_output_ntc;
#if defined USE_GPU && ((TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4) || TORCH_VERSION_MAJOR >= 3)
    if (model_stats->use_flash) {
        float softmax_scale = 1.0 / std::sqrt(head_dim);

        auto qkv_chunks = qkv.chunk(3, 2);
        auto q = qkv_chunks[0].squeeze(2);
        auto k = qkv_chunks[1].squeeze(2);
        auto v = qkv_chunks[2].squeeze(2);
        
        auto flash_res = at::_flash_attention_forward(
            q, k, v,
            std::nullopt, std::nullopt, // cum_seq_qk
            qkv.size(1), qkv.size(1), // size_qk
            0.0, // dropout
            false, // casual
            false, // return debug mask
            softmax_scale,
            win_lower,
            win_upper,
            std::nullopt, // seqused k
            std::nullopt // alibi slopes
        );
        attn_output_ntc = std::get<0>(flash_res).reshape({N, T, C});
    } else
#endif
    {
        qkv = qkv.permute({2, 0, 3, 1, 4}); // N T 3 H D -> 3 N H T D
        attn_output_ntc = torch::empty({N, T, C}, x.options());
        auto attn_window_mask = get_attn_window_mask(T);
        auto attn_output = attn_output_ntc.view({N, T, nhead, head_dim}).transpose(1, 2);
        // // The MPS backend refuses to work on a span of the mask that doesn't have an
        // // alignment of 4 elements, so pad the amount we process each loop to that.
        const auto elems_per_split = pad_to(div_round_up(T, int64_t{num_splits}), int64_t{4});
        for (int i = 0; i < num_splits; ++i) {
            const auto qb = i * elems_per_split;
            if (qb >= T) {
                break;
            }
            const auto qe = std::min(T, qb + elems_per_split);
            const auto kvb = std::max<int64_t>(0, qb - win_lower);
            const auto kve = std::min<int64_t>(T, qe + win_upper);
            const auto q = qkv[0].slice(-2, qb, qe);
            const auto k = qkv[1].slice(-2, kvb, kve);
            const auto v = qkv[2].slice(-2, kvb, kve);
            const auto mask = attn_window_mask.index({Slice(qb, qe), Slice(kvb, kve)});
            c10::optional<torch::Tensor> opt_mask;
            // Not using the mask gets us significantly better performance, at the cost of some
            // accuracy. Accuracy loss is minimised by larger num_splits.
            opt_mask = mask;
            attn_output.slice(-2, qb, qe) = torch::scaled_dot_product_attention(q, k, v, opt_mask);
        }
    }

    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_sdp_attn += b-a;
#ifdef OPENFISH_DUMP
    dump_tensor("attn_window_mask", get_attn_window_mask(T), false);
    dump_tensor("attn_sdpa_out", attn_output_ntc);
    dump_tensor("attn_out_proj_weight", out_proj->weight, false);
    dump_tensor("attn_out_proj_bias", out_proj->bias, false);
#endif

    a = realtime();
    x = at::linear(attn_output_ntc, out_proj->weight, out_proj->bias);
#ifdef OPENFISH_DUMP
    dump_tensor("attn_out_proj_out", x);
#endif
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_out_proj += b-a;

    return x;
};

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params_, const torch::TensorOptions &options, tx_stats_t *_model_stats) : params(params_) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false, true, params.attn_window, options, _model_stats));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));
    model_stats = _model_stats;

    const torch::Tensor deepnorm_alpha = torch::tensor(params.deepnorm_alpha);
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

torch::Tensor TxEncoderImpl::forward(torch::Tensor x) {
    torch::Tensor attn, f;
    const auto deepnorm_alpha = named_buffers()["deepnorm_alpha"];
    double a, b;

    auto run_norm = [&](RMSNorm &norm, const torch::Tensor &in, at::Tensor &weight) {
#if defined USE_GPU
        auto MN = in.size(0) * in.size(1);
        auto output = torch::empty({in.size(0), in.size(1), in.size(2)}, in.options());
        auto K = in.size(2);
        auto eps = 1e-5f;
        
        openfish_rmsnorm_gpu(
            in.contiguous().data_ptr(),
            x.contiguous().data_ptr(),
            weight.contiguous().data_ptr(),
            output.contiguous().data_ptr(),
            MN,
            K,
            deepnorm_alpha.flatten()[0].item<float>(),
            eps
        );
        x = output;
#else
        auto k = in + (x * deepnorm_alpha);
        x = norm(k);
#endif
    };

#ifdef OPENFISH_DUMP
    dump_layer = dump_encoder_calls < dump_env_int("OPENFISH_DUMP_LAYERS", 1) ? (int)dump_encoder_calls : -1;
    ++dump_encoder_calls;
    dump_tensor("enc_in", x);
    dump_tensor("norm_deepnorm_alpha", deepnorm_alpha, false);
    dump_tensor("norm1_weight", norm1->weight, false);
    dump_tensor("norm2_weight", norm2->weight, false);
#endif

    a = realtime();
    attn = self_attn(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_self_attn += b-a;

#ifdef OPENFISH_DUMP
    dump_tensor("norm1_in", attn);
    dump_tensor("norm1_residual", x);
#endif
    a = realtime();
    run_norm(norm1, attn, norm1->weight);
#ifdef OPENFISH_DUMP
    dump_tensor("norm1_out", x);
#endif
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_norm1 += b-a;

    a = realtime();
    f = ff(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_ff += b-a;

#ifdef OPENFISH_DUMP
    dump_tensor("norm2_in", f);
    dump_tensor("norm2_residual", x);
#endif
    a = realtime();
    run_norm(norm2, f, norm2->weight);
#ifdef OPENFISH_DUMP
    dump_tensor("norm2_out", x);
#endif
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_norm2 += b-a;

#ifdef OPENFISH_DUMP
    if (dump_layer >= 0) {
        dump_tensor("enc_out", x);
        if (dump_layer + 1 == dump_env_int("OPENFISH_DUMP_LAYERS", 1) && std::getenv("OPENFISH_DUMP_EXIT")) {
            std::fprintf(stderr, "[OPENFISH_DUMP] dumped %d layer(s) to %s; exiting\n", dump_layer + 1, std::getenv("OPENFISH_DUMP_DIR"));
            std::fflush(nullptr);
            std::_Exit(0);
        }
    }
    dump_layer = -1;
#endif
    return x;
}

TxEncoderStackImpl::TxEncoderStackImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats) {
    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        TxEncoder encoder(params, options, model_stats);
        stack->push_back(register_module("transformer_encoder" + std::to_string(i), encoder));
    }
};

torch::Tensor TxEncoderStackImpl::forward(const torch::Tensor &x) {
    return stack->forward(x);
}

LinearUpsampleImpl::LinearUpsampleImpl(const EncoderUpsampleParams &params) : scale_factor(params.scale_factor) {
    linear = register_module("linear", Linear(LinearOptions(params.d_model, scale_factor * params.d_model).bias(true)));
};

torch::Tensor LinearUpsampleImpl::forward(const torch::Tensor &x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    torch::Tensor out = linear(x).reshape({N, scale_factor * T, C});
    return out;
};

LinearScaledCRFImpl::LinearScaledCRFImpl(const CRFEncoderParams &params) {
    m_params = params;
    linear = register_module("linear", Linear(LinearOptions(m_params.insize, m_params.outsize()).bias(false)));
};

torch::Tensor LinearScaledCRFImpl::forward(const torch::Tensor &x) {
    if (!scale_applied) {
        linear->weight *= m_params.scale;
        scale_applied = true;
    }
    return linear(x);
}

TxModelImpl::TxModelImpl(const CRFModelConfig &config, const torch::TensorOptions &options, tx_stats_t *_model_stats) : m_options(options) {
    convs = register_module("convs", ::ConvStack(config.convs));
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config.tx->tx, m_options, _model_stats));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(config.tx->upsample));
    crf = register_module("crf", LinearScaledCRF(config.tx->crf));
    model_stats = _model_stats;

}

torch::Tensor TxModelImpl::forward(const torch::Tensor &x) {
    torch::Tensor h;
    double a, b;

    a = realtime();
    h = convs->forward(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_conv_stack += b-a;
    
    a = realtime();
    h = tx_encoder(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_tx_encoder += b-a;

    a = realtime();
    h = tx_decoder(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_tx_decoder += b-a;

    a = realtime();
    h = crf(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_crf += b-a;

    // Returns: NTC
    return h;
}

std::vector<torch::Tensor> load_tx_model_weights(const std::string &dir) {
    auto tensors = std::vector<std::string>{
            // convs 0-4
            "conv.0.conv.weight.tensor",
            "conv.0.conv.bias.tensor",
            "conv.1.conv.weight.tensor",
            "conv.1.conv.bias.tensor",
            "conv.2.conv.weight.tensor",
            "conv.2.conv.bias.tensor",
            "conv.3.conv.weight.tensor",
            "conv.3.conv.bias.tensor",
            "conv.4.conv.weight.tensor",
            "conv.4.conv.bias.tensor",

            // tx encoder layer 0
            "transformer_encoder.0.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.bias.tensor",
            "transformer_encoder.0.ff.fc1.weight.tensor",
            "transformer_encoder.0.ff.fc2.weight.tensor",
            "transformer_encoder.0.norm1.weight.tensor",
            "transformer_encoder.0.norm2.weight.tensor",

            // tx encoder layer 1
            "transformer_encoder.1.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.bias.tensor",
            "transformer_encoder.1.ff.fc1.weight.tensor",
            "transformer_encoder.1.ff.fc2.weight.tensor",
            "transformer_encoder.1.norm1.weight.tensor",
            "transformer_encoder.1.norm2.weight.tensor",

            // tx encoder layer 2
            "transformer_encoder.2.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.bias.tensor",
            "transformer_encoder.2.ff.fc1.weight.tensor",
            "transformer_encoder.2.ff.fc2.weight.tensor",
            "transformer_encoder.2.norm1.weight.tensor",
            "transformer_encoder.2.norm2.weight.tensor",

            // tx encoder layer 3
            "transformer_encoder.3.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.bias.tensor",
            "transformer_encoder.3.ff.fc1.weight.tensor",
            "transformer_encoder.3.ff.fc2.weight.tensor",
            "transformer_encoder.3.norm1.weight.tensor",
            "transformer_encoder.3.norm2.weight.tensor",

            // tx encoder layer 4
            "transformer_encoder.4.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.bias.tensor",
            "transformer_encoder.4.ff.fc1.weight.tensor",
            "transformer_encoder.4.ff.fc2.weight.tensor",
            "transformer_encoder.4.norm1.weight.tensor",
            "transformer_encoder.4.norm2.weight.tensor",

            // tx encoder layer 5
            "transformer_encoder.5.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.bias.tensor",
            "transformer_encoder.5.ff.fc1.weight.tensor",
            "transformer_encoder.5.ff.fc2.weight.tensor",
            "transformer_encoder.5.norm1.weight.tensor",
            "transformer_encoder.5.norm2.weight.tensor",

            // tx encoder layer 6
            "transformer_encoder.6.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.bias.tensor",
            "transformer_encoder.6.ff.fc1.weight.tensor",
            "transformer_encoder.6.ff.fc2.weight.tensor",
            "transformer_encoder.6.norm1.weight.tensor",
            "transformer_encoder.6.norm2.weight.tensor",

            // tx encoder layer 7
            "transformer_encoder.7.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.bias.tensor",
            "transformer_encoder.7.ff.fc1.weight.tensor",
            "transformer_encoder.7.ff.fc2.weight.tensor",
            "transformer_encoder.7.norm1.weight.tensor",
            "transformer_encoder.7.norm2.weight.tensor",

            // tx encoder layer 8
            "transformer_encoder.8.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.bias.tensor",
            "transformer_encoder.8.ff.fc1.weight.tensor",
            "transformer_encoder.8.ff.fc2.weight.tensor",
            "transformer_encoder.8.norm1.weight.tensor",
            "transformer_encoder.8.norm2.weight.tensor",

            // tx encoder layer 9
            "transformer_encoder.9.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.bias.tensor",
            "transformer_encoder.9.ff.fc1.weight.tensor",
            "transformer_encoder.9.ff.fc2.weight.tensor",
            "transformer_encoder.9.norm1.weight.tensor",
            "transformer_encoder.9.norm2.weight.tensor",

            // tx encoder layer 10
            "transformer_encoder.10.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.bias.tensor",
            "transformer_encoder.10.ff.fc1.weight.tensor",
            "transformer_encoder.10.ff.fc2.weight.tensor",
            "transformer_encoder.10.norm1.weight.tensor",
            "transformer_encoder.10.norm2.weight.tensor",

            // tx encoder layer 11
            "transformer_encoder.11.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.bias.tensor",
            "transformer_encoder.11.ff.fc1.weight.tensor",
            "transformer_encoder.11.ff.fc2.weight.tensor",
            "transformer_encoder.11.norm1.weight.tensor",
            "transformer_encoder.11.norm2.weight.tensor",

            // tx encoder layer 12
            "transformer_encoder.12.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.bias.tensor",
            "transformer_encoder.12.ff.fc1.weight.tensor",
            "transformer_encoder.12.ff.fc2.weight.tensor",
            "transformer_encoder.12.norm1.weight.tensor",
            "transformer_encoder.12.norm2.weight.tensor",

            // tx encoder layer 13
            "transformer_encoder.13.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.bias.tensor",
            "transformer_encoder.13.ff.fc1.weight.tensor",
            "transformer_encoder.13.ff.fc2.weight.tensor",
            "transformer_encoder.13.norm1.weight.tensor",
            "transformer_encoder.13.norm2.weight.tensor",

            // tx encoder layer 14
            "transformer_encoder.14.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.bias.tensor",
            "transformer_encoder.14.ff.fc1.weight.tensor",
            "transformer_encoder.14.ff.fc2.weight.tensor",
            "transformer_encoder.14.norm1.weight.tensor",
            "transformer_encoder.14.norm2.weight.tensor",

            // tx encoder layer 15
            "transformer_encoder.15.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.bias.tensor",
            "transformer_encoder.15.ff.fc1.weight.tensor",
            "transformer_encoder.15.ff.fc2.weight.tensor",
            "transformer_encoder.15.norm1.weight.tensor",
            "transformer_encoder.15.norm2.weight.tensor",

            // tx encoder layer 16
            "transformer_encoder.16.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.bias.tensor",
            "transformer_encoder.16.ff.fc1.weight.tensor",
            "transformer_encoder.16.ff.fc2.weight.tensor",
            "transformer_encoder.16.norm1.weight.tensor",
            "transformer_encoder.16.norm2.weight.tensor",

            // tx encoder layer 17
            "transformer_encoder.17.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.bias.tensor",
            "transformer_encoder.17.ff.fc1.weight.tensor",
            "transformer_encoder.17.ff.fc2.weight.tensor",
            "transformer_encoder.17.norm1.weight.tensor",
            "transformer_encoder.17.norm2.weight.tensor",

            // tx decoder
            "upsample.linear.weight.tensor",
            "upsample.linear.bias.tensor",

            // linear CRF
            "crf.linear.weight.tensor",
    };

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, int nthreads) {
    if (model_stats) {
        model_stats->use_flash = use_flash;
        model_stats->nthreads = nthreads;
    }
    auto model = TxModel(model_config, options, model_stats);
    auto state_dict = load_tx_model_weights(model_config.model_path);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    if (use_flash) {
        INFO("%s", "flash attention enabled");
    } else {
        INFO("%s", "flash attention disabled");
    }

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}
