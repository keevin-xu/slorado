#include <math.h>
#include <string>

#include "CRFModel.h"
#include "error.h"
#include "misc.h"
#include "tensor_chunk_utils.h"

#if defined(USE_GPU) || defined(HAVE_NPU)
#include <openfish/openfish.h>
#endif

#ifdef HAVE_NPU
#include <algorithm>
#include <mutex>
#include <unordered_map>
#endif

using namespace torch::nn;

#ifdef HAVE_NPU
bool npu_gemm_out_f32();  // TxModel.cpp: OPENFISH_NPU_GEMM_OUT=f32

// swish on the NPU (mlir-air silu, op "swish"): bf16 on the host, back into x's device, dtype and memory layout.
static torch::Tensor silu_npu(const torch::Tensor &x) {
    auto xb = x.contiguous().to(torch::kBFloat16).to(torch::kCPU).contiguous();
    auto ob = torch::empty_like(xb);
    openfish_silu_npu_bf16(reinterpret_cast<const uint16_t *>(xb.data_ptr()), reinterpret_cast<uint16_t *>(ob.data_ptr()),
                           xb.numel());
    auto out = torch::empty_like(x);
    out.copy_(ob.to(x.device()));
    return out;
}

// Conv1d on the NPU as im2col + the openfish NPU GEMM (OPENFISH_NPU_OPS "conv"). The fused-cast GEMM ELF
// needs K % 256 == 0 and N % 512 == 0, so patches and the (cached, host fp32) weight are zero-padded.
static torch::Tensor conv1d_npu(torch::nn::Conv1d &conv, const torch::Tensor &x, int winlen, int stride) {
    const auto &w = conv->weight;  // [Cout, Cin, k]
    const int64_t Cout = w.size(0), K = w.size(1) * w.size(2), k = w.size(2);
    const int64_t Kp = (K + 255) / 256 * 256, Np = (Cout + 511) / 512 * 512;
    static std::mutex mutex;
    static std::unordered_map<const void *, torch::Tensor> wcache;
    torch::Tensor wp;
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = wcache.find(w.data_ptr());
        if (it == wcache.end()) {
            auto t = torch::zeros({Np, Kp}, torch::TensorOptions().dtype(torch::kFloat32));
            t.narrow(0, 0, Cout).narrow(1, 0, K).copy_(w.reshape({Cout, K}).to(torch::kCPU, torch::kFloat32));
            it = wcache.emplace(w.data_ptr(), t).first;
        }
        wp = it->second;
    }
    const int64_t N = x.size(0);
    auto patches = torch::constant_pad_nd(x, {winlen / 2, winlen / 2}).unfold(2, k, stride).permute({0, 2, 1, 3});
    const int64_t Tout = patches.size(1);  // [N, Tout, Cin, k]
    auto out = torch::empty({N, Tout, Cout}, x.options());
    const int64_t slice = 8;  // chunks per NPU call: bounds the host fp32 im2col buffer
    const bool host = x.device().is_cpu() && x.scalar_type() == torch::kFloat32;
    for (int64_t b = 0; b < N; b += slice) {
        const int64_t nb = std::min(slice, N - b);
        if (!host) {
            // iGPU path: pad + cast to bf16 on the GPU, copy 2-byte values, NPU, copy back, cast on the GPU
            auto ab = patches.narrow(0, b, nb).reshape({nb * Tout, K});
            if (Kp != K) {
                ab = torch::constant_pad_nd(ab, {0, Kp - K});
            }
            ab = ab.to(torch::kBFloat16).to(torch::kCPU).contiguous();
            torch::Tensor ob;
            if (npu_gemm_out_f32()) {
                ob = torch::empty({nb * Tout, Np}, ab.options().dtype(torch::kFloat32));
                openfish_linear_npu_bf16_f32("conv", reinterpret_cast<const uint16_t *>(ab.data_ptr()), ob.data_ptr<float>(),
                                             wp.data_ptr<float>(), nb * Tout, Kp, Np);
            } else {
                ob = torch::empty({nb * Tout, Np}, ab.options());
                openfish_linear_npu_bf16("conv", reinterpret_cast<const uint16_t *>(ab.data_ptr()),
                                         reinterpret_cast<uint16_t *>(ob.data_ptr()), wp.data_ptr<float>(), nb * Tout, Kp, Np);
            }
            out.narrow(0, b, nb).copy_(ob.narrow(1, 0, Cout).reshape({nb, Tout, Cout}).to(x.device()));
            continue;
        }
        auto a = patches.narrow(0, b, nb).reshape({nb * Tout, K}).to(torch::kCPU, torch::kFloat32);
        if (Kp != K) {
            a = torch::constant_pad_nd(a, {0, Kp - K});
        }
        a = a.contiguous();
        auto o = torch::empty({nb * Tout, Np}, a.options());
        openfish_linear_npu("conv", a.data_ptr<float>(), o.data_ptr<float>(), wp.data_ptr<float>(), nb * Tout, Kp, Np);
        out.narrow(0, b, nb).copy_(o.narrow(1, 0, Cout).reshape({nb, Tout, Cout}));
    }
    if (conv->bias.defined()) {
        out.add_(conv->bias);
    }
    return out.permute({0, 2, 1});  // [N, Cout, Tout] like Conv1d
}
#endif

ConvStackImpl::ConvStackImpl(const std::vector<ConvParams> &layer_params) {
    for (size_t i = 0; i < layer_params.size(); ++i) {
        layers.emplace_back(layer_params[i]);
        auto &layer = layers.back();
        auto opts = Conv1dOptions(layer.params.insize, layer.params.size, layer.params.winlen)
            .stride(layer.params.stride)
            .padding(layer.params.winlen / 2);
        layer.conv = register_module(std::string("conv") + std::to_string(i + 1), Conv1d(opts));
    }
}

torch::Tensor ConvStackImpl::forward(torch::Tensor x) {
    // Input x is [N, C_in, T_in], contiguity optional
    for (auto &layer : layers) {
#ifdef HAVE_NPU
        if (openfish_npu_op_enabled("conv")) {
            x = conv1d_npu(layer.conv, x, layer.params.winlen, layer.params.stride);
        } else
#endif
        x = layer.conv(x);
        if (layer.params.activation == Activation::SWISH || layer.params.activation == Activation::SWISH_CLAMP) {
#ifdef HAVE_NPU
            if (openfish_npu_op_enabled("swish") && !(x.device().is_cpu() && x.scalar_type() == torch::kFloat32)) {
                x = silu_npu(x);
            } else
#endif
            torch::silu_(x);
            if (layer.params.activation == Activation::SWISH_CLAMP) {
                x.clamp_(c10::nullopt, 3.5f);
            }
        } else if (layer.params.activation == Activation::TANH) {
            x.tanh_();
        } else {
            ERROR("%s", "Unrecognised activation function id.");
        }
    }
    // Output is [N, T_out, C_out], non-contiguous
    return x.transpose(1, 2);
}

ConvStackImpl::ConvLayer::ConvLayer(const ConvParams &conv_params) : params(conv_params) {}

LinearCRFImpl::LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale) : bias(bias_) {
    linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
    if (tanh_and_scale) {
        activation = register_module("activation", Tanh());
    }
};

torch::Tensor LinearCRFImpl::forward(const torch::Tensor &x) {
    // Input x is [N, T, C], contiguity optional
    auto scores = at::linear(x, linear->weight, linear->bias);
    if (activation) {
        scores = activation(scores) * scale;
    }

    // Output is [N, T, C], contiguous
    return scores;
}

LSTMStackImpl::LSTMStackImpl(int num_layers, int size) : layer_size(size) {
    // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
    const auto lstm_opts = LSTMOptions(size, size).batch_first(true);
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        rnns.emplace_back(register_module(label, LSTM(lstm_opts)));
    }
};

torch::Tensor LSTMStackImpl::forward(torch::Tensor x) {
    // Input is [N, T, C], contiguity optional
    for (auto &rnn : rnns) {
        x = std::get<0>(rnn(x.flip(1)));
    }

    // Output is [N, T, C], contiguous
    return (rnns.size() & 1) ? x.flip(1) : x;
}

FLSTMLayerImpl::FLSTMLayerImpl(int C, int K, lstm_stats_t *model_stats) : C_(C), K_(K), model_stats_(model_stats) {
    dn_weight_ih_ = register_parameter("dn_weight_ih", torch::empty({K, C}));
    dn_weight_hh_ = register_parameter("dn_weight_hh", torch::empty({K, C}));
    up_weight_ih_ = register_parameter("up_weight_ih", torch::empty({4 * C, K}));
    up_weight_hh_ = register_parameter("up_weight_hh", torch::empty({4 * C, K}));
    up_bias_ih_   = register_parameter("up_bias_ih",   torch::empty({4 * C}));
    up_bias_hh_   = register_parameter("up_bias_hh",   torch::empty({4 * C}));
}

torch::Tensor FLSTMLayerImpl::forward(torch::Tensor x) {
    // x is [N, T, C]
    x = x.transpose(0, 1).contiguous();  // [T, N, C]
    const int T = x.size(0);
    const int N = x.size(1);
    const bool on_gpu = !x.device().is_cpu();
    double a, b;

    // Cache the transposed hh weight once outside the loop.
    auto up_w_hh_t = up_weight_hh_.t().contiguous();  // [K, 4*C]

    // --- IH precompute: two matmuls over the full sequence ---
    a = realtime();
    auto x_flat = x.view({T * N, x.size(2)});

    // dn_ih: [T*N, C] @ [C, K] -> [T*N, K]
    auto dn_ih = at::linear(x_flat, dn_weight_ih_);

    // ih: [T*N, K] @ [K, 4*C] + bias -> [T, N, 4*C]
    auto ih = at::linear(dn_ih, up_weight_ih_, up_bias_ih_).view({T, N, 4 * C_});
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_flstm_precompute += b - a;

    auto hh = torch::empty({T + 1, N, C_}, x.options());
    hh[0].zero_();
    auto c        = torch::zeros({N, C_},     x.options());
    auto scratch  = torch::empty({N, 4 * C_}, x.options());
    auto dn_hh_buf = torch::empty({N, K_},   x.options());

    a = realtime();
    for (int t = 0; t < T; ++t) {
        // Down project hh: [N, C] @ [C, K] -> [N, K]
        torch::mm_out(dn_hh_buf, hh[t], dn_weight_hh_.t());
        // Up project hh: [N, K] @ [K, 4*C] + bias -> [N, 4*C]
        torch::addmm_out(scratch, up_bias_hh_, dn_hh_buf, up_w_hh_t);

        // Fused epilogue: scratch + ih[t] -> gates -> cell update -> hh[t+1]
#ifdef USE_GPU
        openfish_flstm_step_gpu(scratch.data_ptr(), ih[t].data_ptr(),
                                c.data_ptr(), hh[t + 1].data_ptr(), N, C_);
#else
        auto gates = scratch.add(ih[t]).chunk(4, 1);
        auto i = gates[0].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        auto f = gates[1].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        auto g = gates[2].clamp_(-1.f, 1.f);
        auto o = gates[3].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        c = (f * c) + (i * g);
        hh[t + 1] = o * torch::tanh(c);
#endif
    }
    b = realtime();
    model_stats_->time_flstm_recurrence += b - a;

    using namespace torch::indexing;

    // Return [N, T, C]
    return hh.index({Slice(1, None)}).transpose(0, 1).contiguous();
}

FLSTMStackImpl::FLSTMStackImpl(int num_layers, int C, int K, lstm_stats_t *model_stats) {
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        layers_.emplace_back(register_module(label, FLSTMLayer(C, K, model_stats)));
    }
}

torch::Tensor FLSTMStackImpl::forward(torch::Tensor x) {
    // Flip before every layer (alternating directions); odd layer count → flip output
    for (auto &layer : layers_) {
        x = layer->forward(x.flip(1));
    }
    return (layers_.size() & 1) ? x.flip(1) : x;
}


ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

torch::Tensor ClampImpl::forward(torch::Tensor x) {
    if (active) {
        x.clamp_(min, max);
    }
    return x;
}

CRFModelImpl::CRFModelImpl(const CRFModelConfig &config, lstm_stats_t *model_stats) : model_stats_(model_stats) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv));

    if (config.lstm_inner_dim >= 0) {
        // v6.0+ FLSTM model: decomposed linear + tanh in CRF encoder, no clamp
        flstm_rnns = register_module("rnns", FLSTMStack(config.lstm_layers, lstm_size, config.lstm_inner_dim, model_stats));
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, config.bias, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, true));
    } else if (config.has_out_features) {
        // v4.x model with linear decomposition
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x / v5.x model without linear decomposition
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
    } else {
        // Pre-v4 model
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
    }

}

void CRFModelImpl::load_state_dict(const std::vector<torch::Tensor> &weights) {
    module_load_state_dict(*this, weights);
}


torch::Tensor CRFModelImpl::forward(const torch::Tensor &x) {
    const bool on_gpu = !x.device().is_cpu();
    double a, b;
    torch::Tensor h;

    a = realtime();
    h = convs->forward(x);
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_conv_stack += b - a;

    a = realtime();
    if (flstm_rnns) {
        h = flstm_rnns->forward(h);
    } else {
        h = rnns->forward(h);
    }
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_rnns += b - a;

    a = realtime();
    h = linear1->forward(h);
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_crf_1 += b - a;

    if (linear2) {
        a = realtime();
        h = linear2->forward(h);
        if (on_gpu) torch::cuda::synchronize(x.device().index());
        b = realtime();
        model_stats_->time_crf_2 += b - a;
    }

    if (clamp1) {
        a = realtime();
        h = clamp1->forward(h);
        if (on_gpu) torch::cuda::synchronize(x.device().index());
        b = realtime();
        model_stats_->time_clamp += b - a;
    }

    // Output is [N, T, C]
    return h;
}

std::vector<torch::Tensor> load_lstm_model_weights(const CRFModelConfig &config) {
    const auto &dir = config.model_path;
    auto tensors = std::vector<std::string>{
        "0.conv.weight.tensor", "0.conv.bias.tensor",
        "1.conv.weight.tensor", "1.conv.bias.tensor",
        "2.conv.weight.tensor", "2.conv.bias.tensor",
    };

    // RNN layers start at sublayer index 4 (after 3 convs + 1 permute)
    const int rnn_start = 4;

    if (config.lstm_inner_dim >= 0) {
        // FLSTM: 6 parameters per layer (dn_weight_ih/hh, up_weight_ih/hh, up_bias_ih/hh)
        for (int i = 0; i < config.lstm_layers; ++i) {
            auto p = std::to_string(rnn_start + i) + ".rnn.";
            tensors.push_back(p + "dn_weight_ih.tensor");
            tensors.push_back(p + "dn_weight_hh.tensor");
            tensors.push_back(p + "up_weight_ih.tensor");
            tensors.push_back(p + "up_weight_hh.tensor");
            tensors.push_back(p + "up_bias_ih.tensor");
            tensors.push_back(p + "up_bias_hh.tensor");
        }
        // Intermediate linear then CRF linear (both bias=false)
        const int lin_idx = rnn_start + config.lstm_layers;
        tensors.push_back(std::to_string(lin_idx)     + ".linear.weight.tensor");
        tensors.push_back(std::to_string(lin_idx + 1) + ".linear.weight.tensor");
    } else {
        // Standard LSTM: 4 parameters per layer (weight_ih/hh, bias_ih/hh)
        for (int i = 0; i < config.lstm_layers; ++i) {
            auto p = std::to_string(rnn_start + i) + ".rnn.";
            tensors.push_back(p + "weight_ih_l0.tensor");
            tensors.push_back(p + "weight_hh_l0.tensor");
            tensors.push_back(p + "bias_ih_l0.tensor");
            tensors.push_back(p + "bias_hh_l0.tensor");
        }
        const int lin_idx = rnn_start + config.lstm_layers;
        tensors.push_back(std::to_string(lin_idx) + ".linear.weight.tensor");
        if (config.bias) {
            tensors.push_back(std::to_string(lin_idx) + ".linear.bias.tensor");
        }
        if (config.has_out_features) {
            tensors.push_back(std::to_string(lin_idx + 1) + ".linear.weight.tensor");
        }
    }

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
    auto model = CRFModel(model_config, model_stats);
    auto state_dict = load_lstm_model_weights(model_config);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}