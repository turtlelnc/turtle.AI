// Exercise the actual model definitions until they are split from the CLI.
#define main turtle_cli_main
#include "../main.cpp"
#undef main

static nlohmann::json compare_logits(const array& reference, const array& actual) {
    array ref = astype(reference, float32), got = astype(actual, float32);
    array delta = abs(subtract(ref, got));
    const float max_error = max(delta).item<float>();
    const float normalized = max(divide(delta,
        add(array(0.002f), multiply(array(0.001f), abs(ref))))).item<float>();
    return {{"max_abs", max_error}, {"normalized_error", normalized},
            {"passed", std::isfinite(normalized) && normalized <= 1.0f},
            {"rms", sqrt(mean(square(delta))).item<float>()},
            {"reference_max", max(abs(ref)).item<float>()}};
}

int main(int argc, char** argv) {
  try {
    set_default_device(Device(Device::cpu));
    random::seed(7);
    std::string model_path;
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if (arg=="--fast-sdpa") fast_decode_sdpa=true;
        else if (arg=="--gather-sparse") gather_sparse_attention=true;
        else if (arg=="--split-qkv") fused_qkv_projection=false;
        else if (arg=="--generic-rms") fast_rmsnorm=false;
        else if (model_path.empty() && arg.rfind("--",0)!=0) model_path=arg;
        else throw std::invalid_argument("Unknown check option: "+arg);
    }
    nlohmann::json metadata;
    int vocab = 64, dim = 16, loops = 2, window = 4, global = 2;
    int experts = 4, top = 2, wide = 1, context = 16;
    if (!model_path.empty()) {
        std::ifstream in(model_path, std::ios::binary);
        char signature[8]; in.read(signature, 8);
        if (!in || std::string(signature, 8) != "TRTLMX01") throw std::runtime_error("Expected MLX metadata");
        auto length = checkpoint::read_length(in);
        if (!length || length > checkpoint::max_header_size) throw std::runtime_error("Invalid metadata size");
        std::string text(length, '\0'); in.read(text.data(), length);
        metadata = nlohmann::json::parse(text);
        if (metadata.at("model_mode") != "ar") throw std::runtime_error("AR only");
        vocab=metadata.at("vocab_size"); dim=metadata.at("dim"); loops=metadata.at("max_loop");
        window=metadata.at("window_size"); global=metadata.at("global_topk");
        experts=metadata.at("moe_experts"); top=metadata.at("moe_topk");
        wide=metadata.at("wide_blocks"); context=metadata.at("seq_len");
    }
    OpenMythos model(vocab, dim, loops, window, global, experts, top, wide, context);
#ifdef USE_CAPACITY_MOE
    model.set_training(false);
#endif
    if (!model_path.empty()) {
        std::ifstream in(model_path, std::ios::binary);
        mlx_checkpoint::read_metadata(in, {{"backend","mlx-openmythos"},
            {"byte_order",checkpoint::byte_order()}, {"model_mode","ar"},
#ifdef USE_CAPACITY_MOE
            {"capacity_moe",true}
#else
            {"capacity_moe",false}
#endif
        });
        int step; in.read(reinterpret_cast<char*>(&step), sizeof(step));
        if (!model.mp.load(in)) throw std::runtime_error("Parameter load failed");
    }
    nlohmann::json results = nlohmann::json::array();
    for (bool repeat : {false, true}) {
        std::vector<int> ids(context);
        for (int i=0; i<context; ++i) ids[i] = repeat ? 7 : 6+(i*17)%(vocab-6);
        array full = model(array(ids.data(), {context}, int32)); eval(full);
        auto changed=ids;
        for (int i=context/2; i<context; ++i) changed[i]=6+(changed[i]+13)%(vocab-6);
        array altered=model(array(changed.data(), {context}, int32));
        auto causal=compare_logits(slice(full,{0,0},{context/2,vocab}),slice(altered,{0,0},{context/2,vocab}));
        results.push_back({{"check","causality"},{"repeat",repeat},{"metrics",causal}});
        RecurrentKVCache cache(wide,loops);
        std::vector<array> cached;
        for (int token: ids) {
            array row=model.decode(array(&token,{1},int32),cache,context);eval(row);cached.push_back(row);
        }
        results.push_back({{"check","kv_full_prefix"},{"repeat",repeat},
                           {"metrics",compare_logits(full,concatenate(cached,0))}});
        const int bound = std::min(8, context/2);
        RecurrentKVCache rolling(wide,loops);
        float worst = 0.0f, normalized = 0.0f;
        for (int i=0; i<bound+5; ++i) {
            int token=ids[i%context];
            array row=model.decode(array(&token,{1},int32),rolling,bound);
            std::vector<int> history;
            for (int j=std::max(0,i-bound+1); j<=i; ++j) history.push_back(ids[j%context]);
            int length=history.size();
            array ref=model(array(history.data(),{length},int32));
            auto metric=compare_logits(slice(ref,{length-1,0},{length,vocab}),row);
            worst=std::max(worst,metric["max_abs"].get<float>());
            normalized=std::max(normalized,metric["normalized_error"].get<float>());
        }
        results.push_back({{"check","kv_rolling_context"},{"repeat",repeat},
            {"metrics",{{"max_abs",worst},{"normalized_error",normalized},{"passed",normalized<=1.0f}}}});
        bool rejected=false;
        try { int token=7;model.decode(array(&token,{1},int32),rolling,bound+1); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("Context changes did not require a cache reset");
    }
    // Make all sparse-attention scores exactly equal, with different values.
    // This reproduces the old exact-k decode versus >=threshold forward bug.
    ModelParams attention_params;
    SparseGlobalAttention attention(16,8,2,attention_params);
    attention.Wqkv.W=concatenate({zeros({16,32},float16),
        slice(attention.Wqkv.W,{0,32},{16,48})},1);
    attention.Wqkv.b=zeros({48},float16);
    array inputs=astype(reshape(arange(96),{6,16}),float16);
    array reference=attention(inputs);
    AttentionKVCache attention_cache;
    std::vector<array> attention_rows;
    for (int i=0;i<6;++i)
        attention_rows.push_back(attention.decode(slice(inputs,{i,0},{i+1,16}),attention_cache,6));
    results.push_back({{"check","sparse_attention_ties"},
        {"metrics",compare_logits(reference,concatenate(attention_rows,0))}});
    std::vector<int> batch(context);
    for(int i=0;i<context;++i) batch[i]=6+(i*11)%(vocab-6);
    const int half=context/2;
    array first=model(array(batch.data(),{half},int32));
    array second=model(array(batch.data()+half,{half},int32));
    eval({first,second});
    packed_batch_size=2;
    array together=model(array(batch.data(),{context},int32));
    results.push_back({{"check","packed_sequence_isolation"},
        {"metrics",compare_logits(concatenate({first,second},0),together)}});
    eval(together);
    for(int i=half;i<context;++i) batch[i]=7;
    array changed_batch=model(array(batch.data(),{context},int32));
    results.push_back({{"check","packed_cross_sequence_causality"},
        {"metrics",compare_logits(slice(together,{0,0},{half,vocab}),
                                  slice(changed_batch,{0,0},{half,vocab}))}});
    packed_batch_size=1;
    std::cout << "CHECKS " << results.dump() << std::endl;
    for (const auto& result:results)
        if (!result["metrics"]["passed"].get<bool>()) return 1;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
