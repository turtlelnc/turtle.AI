#include <mlx/mlx.h>
#include <mlx/memory.h> 
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <functional>
#include <numeric> 
#include <cstring>   // 核心修复：用于底层内存强制同步拷贝
#include "BPE.h"
#include <thread>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <deque>
#include <sstream>
#include <optional>
#include <tuple>
#include <limits>
#include <memory>
#include <cstdint>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#include "data.h"
#include "checkpoint_io.h"

using namespace mlx::core;
using namespace bpe;

// ============================================================================
// 🌟 护栏与超参数
// ============================================================================
constexpr float SNR_THRESHOLD   = 1.0f;
constexpr float THRUST_STRENGTH = 0.1f;
constexpr float THRUST_MAX      = 1.0f;

std::mt19937 rng(42);

// ============================================================================
// 🌟 自适应优化器引擎：内存充足用纯内存，内存紧张自动切换 SSD 交换
// ============================================================================

// 查询系统总物理内存（macOS sysctl）
static size_t get_hw_memsize() {
    size_t result = 0;
#ifdef __APPLE__
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) != 0 || memsize <= 0)
        throw std::runtime_error("Cannot query physical memory");
    result = static_cast<size_t>(memsize);
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) throw std::runtime_error("Cannot query physical memory");
    result = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    std::ifstream limit_file("/sys/fs/cgroup/memory.max");
    std::string limit;
    if (limit_file >> limit && limit != "max") result = std::min(result, static_cast<size_t>(std::stoull(limit)));
#endif
    return result;
}
static std::string optimizer_memory_mode = "auto";

// 纯内存模式的优化器状态池（内存充足时使用）
struct TensorBuffer {
    std::vector<float> packed; // 连续布局：[m | v | master_w]，各占 elements 个 float
    std::vector<array> state; // Resident m, v, master, optional P; packed refreshed on save.
};
static std::unordered_map<size_t, TensorBuffer> g_memory_pool;
static bool resident_optimizer = true;
static bool fused_moe_aux = true;
static bool fast_rmsnorm = true;
static bool batch_galore_probes = true;
static bool partition_moe_topk = true;
static bool incremental_decode_active = false;
static bool gather_sparse_attention = false;
static bool fused_qkv_projection = true;
static bool fast_decode_sdpa = false;
static bool vectorized_sparse_decode = true;
static bool bidirectional_training_attention = false;
static int packed_batch_size = 1;
static int rope_empty_prefix_tokens = 0;
static constexpr uint32_t ELF_CHECKPOINT_MAGIC = 0x454C4632;  // "ELF2"
static constexpr uint32_t ELF_T5_CHECKPOINT_MAGIC = 0x454C4633;  // "ELF3"
static constexpr uint32_t ELF_T5_MODE_CHECKPOINT_MAGIC = 0x454C4634;  // "ELF4"
static constexpr uint32_t ELF_DECODER_CHECKPOINT_MAGIC = 0x454C4635;  // "ELF5"
static constexpr uint32_t ELF_GUIDED_CHECKPOINT_MAGIC = 0x454C4636;  // "ELF6"
static constexpr uint32_t ELF_PREFIX_CHECKPOINT_MAGIC = 0x454C4637;  // "ELF7"
static constexpr int ELF_TIME_TOKENS = 4;
static constexpr int ELF_CFG_TOKENS = 4;
static constexpr int ELF_MODE_TOKENS = 4;
static constexpr int ELF_CONDITION_TOKENS =
    ELF_TIME_TOKENS + ELF_CFG_TOKENS + ELF_MODE_TOKENS;
static constexpr int ELF_TIME_EMBED_DIM = 256;
static constexpr int ELF_BOTTLENECK_DIM = 128;
static constexpr uint32_t T5_LATENT_FILE_MAGIC = 0x5435454C;  // "T5EL"
static constexpr uint32_t T5_UNEMBED_FILE_MAGIC = 0x45553554;  // "T5UE"

struct T5LatentDataset {
    std::string path;
    mutable std::ifstream stream;
    uint32_t version = 0;
    uint32_t records = 0;
    uint32_t seq_len = 0;
    uint32_t latent_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t pad_token_id = 0;
    std::streamoff header_bytes = 7 * sizeof(uint32_t);

    explicit T5LatentDataset(const std::string& file_path)
        : path(file_path), stream(path, std::ios::binary) {
        uint32_t magic = 0;
        stream.seekg(0, std::ios::end);
        const std::streamoff file_size = stream.tellg();
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(&records), sizeof(records));
        stream.read(reinterpret_cast<char*>(&seq_len), sizeof(seq_len));
        stream.read(reinterpret_cast<char*>(&latent_dim), sizeof(latent_dim));
        stream.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        stream.read(reinterpret_cast<char*>(&pad_token_id), sizeof(pad_token_id));
        if (!stream || magic != T5_LATENT_FILE_MAGIC || version != 1 ||
            records == 0 || seq_len == 0 || latent_dim == 0 || vocab_size == 0 || pad_token_id >= vocab_size ||
            seq_len > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            latent_dim > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            vocab_size > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid T5 latent dataset: " + path);
        const uint64_t bytes = uint64_t(seq_len) * (sizeof(int32_t) + uint64_t(latent_dim) * sizeof(float16_t));
        if (file_size < header_bytes || bytes > uint64_t(file_size - header_bytes) / records ||
            bytes * records != uint64_t(file_size - header_bytes))
            throw std::invalid_argument("T5 latent dataset size mismatch: " + path);
    }

    size_t record_bytes() const {
        return static_cast<size_t>(seq_len) * sizeof(int32_t) +
               static_cast<size_t>(seq_len) * latent_dim * sizeof(float16_t);
    }

    void read_record(uint32_t index, int32_t* ids, float16_t* latents) const {
        if (index >= records) throw std::out_of_range("T5 latent record index");
        stream.clear();
        stream.seekg(header_bytes + static_cast<std::streamoff>(index * record_bytes()));
        stream.read(reinterpret_cast<char*>(ids), seq_len * sizeof(int32_t));
        stream.read(reinterpret_cast<char*>(latents),
                static_cast<std::streamsize>(seq_len) * latent_dim * sizeof(float16_t));
        if (!stream) throw std::runtime_error("Failed reading T5 latent record");
        for (uint32_t i = 0; i < seq_len; ++i)
            if (ids[i] < 0 || static_cast<uint32_t>(ids[i]) >= vocab_size)
                throw std::runtime_error("T5 latent token ID is out of range");
        for (size_t i = 0; i < static_cast<size_t>(seq_len) * latent_dim; ++i)
            if (!std::isfinite(static_cast<float>(latents[i]))) throw std::runtime_error("T5 latents contain NaN/Inf");
    }
};

struct T5Unembedding {
    uint32_t vocab_size = 0;
    uint32_t latent_dim = 0;
    array weights;

    explicit T5Unembedding(const std::string& path)
        : weights(array(0.0f, float16)) {
        std::ifstream in(path, std::ios::binary);
        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0);
        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        in.read(reinterpret_cast<char*>(&latent_dim), sizeof(latent_dim));
        if (!in || magic != T5_UNEMBED_FILE_MAGIC || version != 1 ||
            vocab_size == 0 || latent_dim == 0 ||
            vocab_size > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            latent_dim > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid T5 unembedding file: " + path);
        const uint64_t elements = uint64_t(vocab_size) * latent_dim;
        if (file_size < 16 || elements > uint64_t(file_size - 16) / sizeof(float16_t) ||
            elements * sizeof(float16_t) != uint64_t(file_size - 16))
            throw std::invalid_argument("T5 unembedding size mismatch: " + path);
        std::vector<float16_t> host(
            static_cast<size_t>(vocab_size) * latent_dim);
        in.read(reinterpret_cast<char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float16_t)));
        if (!in) throw std::runtime_error("Failed reading T5 unembedding table");
        for (auto value : host) if (!std::isfinite(static_cast<float>(value))) throw std::runtime_error("T5 unembedding contains NaN/Inf");
        weights = array(host.data(), {
            static_cast<int>(vocab_size), static_cast<int>(latent_dim)}, float16);
    }
};

// ---- GaLore（梯度低秩投影）逐参数元数据 ----
// 思路来源：Zhao et al. 2024 "GaLore: Memory-Efficient LLM Training by
// Gradient Low-Rank Projection"。核心：权重保持全秩、每步完整更新，
// 只有优化器状态 (m, v) 存在低维子空间里——这是内存节省的全部来源。
// 子空间取自梯度的 top-r 奇异方向（随机 range finder 近似），并且每隔
// refresh 步用当前梯度重建一次。"周期重建"不是可选项而是必需品：独立
// Python 诊断实验证明静态子空间会因为"损失面是弯的、旧方向逐渐失效"
// 而卡在一个明显高于 baseline 的水平；周期重建后质量可追平全量Adam
// （实验数据：静态 val 0.089 / 周期重建 0.061 / 完整GaLore形态 0.0141
//  vs 全量Adam 0.0155，同时优化器内存省 88%）。
struct GaloreMeta {
    bool use = false;          // 该参数是否走 GaLore 低秩投影路径
    int rows = 0, cols = 0;    // 参数 reshape 成 2D 后的形状（cols = 最后一维）
    int r = 0;                 // 实际使用的子空间秩
    bool project_rows = true;  // true: 投影行侧，P=[rows,r]，状态=[r,cols]
                               // false: 投影列侧，P=[cols,r]，状态=[rows,r]
                               // 总是投影较大的一侧，状态存储量最小化
    size_t proj_elems = 0;     // 单个 m（或 v）在子空间里的元素数
    size_t p_elems = 0;        // 投影矩阵 P 的元素数
    size_t packed_floats = 0;  // packed 缓冲总 float 数
    bool p_ready = false;      // P 是否已构建（首步/断点恢复后为 false → 触发重建）
    int steps_in_subspace = 0; // 当前子空间内已走的步数（用于 Adam 偏置修正——
                               // 每次重建后 m,v 清零，修正系数也要从头计）

    // ---- 动态刷新触发（余弦偏移信号）所需状态 ----
    // 独立Python诊断实验证实：固定间隔的刷新在训练早期（损失面变化最剧烈时）
    // 会导致子空间长时间过时，200步实测loss从baseline的3.75劣化到5.26；
    // 改用"梯度方向与构建子空间时的参考梯度的余弦相似度"作为触发信号后，
    // 同等实验条件下前50步loss从0.157改善到0.092，且最终质量不降反升。
    //
    // 优化：g_ref 持久化为 MLX array（GPU 常驻），而不是 host vector 每步
    // 重新拷贝——原实现每个 step 对每个 GaLore 参数都做一次
    // array(g_ref_ptr.data(), ...) host→GPU 全量拷贝（最大的参数 192×16384
    // 就是 12.6MB），加上一次 .item<float>() 同步，13 个参数每步的余弦
    // 检查开销非常可观。现在快照只重建时更新一次，ref_norm 也顺手缓存。
    array g_ref;             // 构建子空间那一步的梯度快照（GPU 常驻，重建时更新）
    float ref_norm = 0.0f;   // ||g_ref||（重建时缓存，避免每步重复全量归约）
    int last_refresh_step = -1;  // 上次刷新发生的全局step，配合 min/max interval 使用

    GaloreMeta() : g_ref(array({0.0f}, float32)) {}  // 默认构造：空标量占位，
                                                     // 避免 array 无默认构造
};

struct ModelParams {
    std::vector<array*> ptrs;
    std::string swap_dir = "./ssd_swap/";
    bool use_swap = false;  // 运行时决策：true = SSD 交换，false = 纯内存

    // ---- GaLore 全局配置：必须在 init_swap() 之前设置（main 从 CLI 读入）----
    bool   galore_enabled  = true;
    // 子空间秩默认保持 128：实测 rank64 在 120 步后收敛明显变慢
    // （同种子 loss diff 从 0.03 扩大到 0.53），rank 是训练质量参数，
    // 不能为了省投影计算而牺牲收敛。若确需省显存可 --galore-rank 64。
    // rank 持久化在 checkpoint 里，断点恢复以 ckpt 存值为准。
    int    galore_rank     = 128;    // 子空间秩
    size_t galore_min_size = 65536;  // 小于该参数量的张量保持原版全量更新

    // ---- 动态刷新（余弦偏移触发）参数 ----
    // galore_refresh 保留原名但改变语义：现在是"最大兜底间隔"——不管余弦
    // 相似度信号是否触发，超过这个步数也强制刷新一次，防止信号本身失效
    // (比如梯度长期恰好在同一方向附近小幅波动，余弦值一直不掉但子空间
    // 其实该刷新了)时子空间无限期不更新的极端情况。
    int    galore_refresh     = 300;   // 最大兜底间隔（原语义"固定间隔"，现在是安全上限）
    int    galore_min_interval = 10;   // 最短间隔：刷新后至少这么多步内不再刷新，
                                       // 防止余弦信号噪声导致刷新过于频繁
    float  galore_cos_thresh  = 0.5f;  // 余弦相似度低于此值时触发刷新
    bool   galore_fixed_refresh = false; // true: 退化为纯固定间隔模式（=galore_refresh），
                                         // 用于和动态版本做A/B对照或问题排查时快速切回
    std::vector<GaloreMeta> gmeta;

    void register_param(array& p) {
        ptrs.push_back(&p);
    }

    // 逐参数决定是否启用 GaLore 并计算 packed 布局。
    // packed 布局：
    //   GaLore 参数: [m_proj | v_proj | master_w(全尺寸FP32) | P矩阵]
    //   普通参数:    [m | v | master_w]（与原版完全一致）
    // 1D 参数（bias、RMSNorm权重）和小张量不投影：它们的优化器状态本来
    // 就小，投影没有收益，反而增加矩阵乘法开销。
    void _build_meta() {
        gmeta.assign(ptrs.size(), GaloreMeta{});
        for (size_t i = 0; i < ptrs.size(); ++i) {
            array& p = *ptrs[i];
            GaloreMeta& g = gmeta[i];
            size_t sz = p.size();
            if (galore_enabled && p.ndim() >= 2 && sz >= galore_min_size) {
                int cols = p.shape(-1);
                int rows = static_cast<int>(sz / cols);
                g.use = true;
                g.rows = rows; g.cols = cols;
                g.r = std::min(galore_rank, std::min(rows, cols));
                g.project_rows = (rows >= cols);
                g.proj_elems = g.project_rows ? (size_t)g.r * cols
                                              : (size_t)rows * g.r;
                g.p_elems = g.project_rows ? (size_t)rows * g.r
                                           : (size_t)cols * g.r;
                g.packed_floats = g.proj_elems * 2 + sz + g.p_elems;
            } else {
                g.packed_floats = sz * 3;
            }
        }
    }

    // master_w 在 packed 中的偏移（GaLore 参数的 m/v 是子空间尺寸，偏移不同）
    size_t _master_offset(size_t i) const {
        return gmeta[i].use ? gmeta[i].proj_elems * 2 : ptrs[i]->size() * 2;
    }

    // 启动时预算检查，自动选择引擎
    void init_swap() {
        // --- 先构建 GaLore 逐参数元数据（决定每个参数的 packed 尺寸）---
        _build_meta();

        // --- 计算优化器总需求 ---
        size_t optimizer_bytes = 0;  // 按实际 packed 布局计算，均为 FP32
        size_t dense_equiv     = 0;  // 若不启用GaLore的等效需求（用于对比展示）
        size_t model_bytes     = 0;  // 模型权重 FP16
        int n_projected = 0;
        for (size_t i = 0; i < ptrs.size(); ++i) {
            optimizer_bytes += gmeta[i].packed_floats * sizeof(float);
            dense_equiv     += ptrs[i]->size() * sizeof(float) * 3;
            model_bytes     += ptrs[i]->nbytes();
            if (gmeta[i].use) ++n_projected;
        }
        if (galore_enabled && n_projected > 0) {
            std::cout << "🛰️  GaLore已启用: " << n_projected << " 个大张量走梯度低秩投影"
                      << " (rank=" << galore_rank << ", refresh=" << galore_refresh << "步)"
                      << "，优化器缓冲 " << dense_equiv / 1e9f << " GB -> "
                      << optimizer_bytes / 1e9f << " GB" << std::endl;
        }
        // 额外预留：MLX 激活值 + macOS 系统占用
        constexpr size_t ACTIVATION_HEADROOM = 1ULL * 1024 * 1024 * 1024;  // 1 GB
        constexpr size_t OS_HEADROOM         = 2ULL * 1024 * 1024 * 1024;  // 2 GB
        size_t total_needed = optimizer_bytes + model_bytes
                            + ACTIVATION_HEADROOM + OS_HEADROOM;

        // 安全上限 = 物理内存 × 60%（为 macOS unified memory 留余量）
        size_t hw_mem    = get_hw_memsize();
        size_t safe_limit = hw_mem * 6 / 10;

        float needed_gb = total_needed / 1e9f;
        float limit_gb  = safe_limit  / 1e9f;
        float hw_gb     = hw_mem      / 1e9f;

        std::cout << "🧮 内存预算：优化器+模型需 " << needed_gb
                  << " GB，物理内存 " << hw_gb
                  << " GB，安全上限 " << limit_gb << " GB" << std::endl;

        use_swap = optimizer_memory_mode == "swap" ||
            (optimizer_memory_mode == "auto" && total_needed > safe_limit);

        if (use_swap) {
            std::cout << "⚠️  内存不足，启用 SSD 交换引擎（swap_dir: "
                      << swap_dir << "）" << std::endl;
            std::filesystem::create_directories(swap_dir);
            _init_ssd();
        } else {
            std::cout << "✅ 内存充足，使用纯内存高速模式。" << std::endl;
            _init_mem();
        }
    }

    std::vector<array> get_values() const {
        std::vector<array> vals;
        for (auto p : ptrs) vals.push_back(*p);
        return vals;
    }

    void set_values(const std::vector<array>& vals) {
        for (size_t i = 0; i < ptrs.size(); ++i) *(ptrs[i]) = vals[i];
    }

    // checkpoint 保存：格式兼容两种模式
    // v2 格式（GaLore）：先写 magic + 影响packed布局的GaLore配置，再逐参数写
    // [权重bytes | packed_floats数 | packed数据]。持久化 rank/min_size/enabled
    // 是为了防止一个坑：如果训练用 rank=256、推理时忘了传参数用默认128，
    // packed 尺寸就对不上，旧逻辑会静默读错数据。现在载入方直接采用
    // checkpoint 里存的布局配置，杜绝这种不一致。
    static constexpr uint32_t CKPT_MAGIC = 0x474C5232;  // "GLR2"
    void save(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
        int32_t g_en = galore_enabled ? 1 : 0;
        int32_t g_rk = galore_rank;
        uint64_t g_ms = galore_min_size;
        os.write(reinterpret_cast<const char*>(&g_en), sizeof(g_en));
        os.write(reinterpret_cast<const char*>(&g_rk), sizeof(g_rk));
        os.write(reinterpret_cast<const char*>(&g_ms), sizeof(g_ms));

        size_t n = ptrs.size();
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));

        // Materialize the entire checkpoint snapshot with one graph evaluation.
        // The previous per-parameter eval (plus another eval for each resident
        // optimizer state) introduced hundreds of CPU/GPU synchronization points.
        std::vector<array> snapshot;
        snapshot.reserve(n * 5);
        for (const auto* p : ptrs) snapshot.push_back(*p);
        if (!use_swap) {
            for (size_t i = 0; i < n; ++i) {
                const auto& state = g_memory_pool.at(i).state;
                snapshot.insert(snapshot.end(), state.begin(), state.end());
            }
        }
        eval(snapshot);

        for (size_t i = 0; i < n; ++i) {
            size_t bytes = ptrs[i]->nbytes();
            os.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
            os.write(reinterpret_cast<const char*>(ptrs[i]->data<char>()), bytes);

            uint64_t pf = gmeta[i].packed_floats;
            os.write(reinterpret_cast<const char*>(&pf), sizeof(pf));
            if (use_swap) {
                // 从 SSD 读回后写入 checkpoint
                std::vector<float> tmp(pf);
                std::ifstream is(swap_dir + std::to_string(i) + ".bin", std::ios::binary);
                if (!is.read(reinterpret_cast<char*>(tmp.data()), pf * sizeof(float)))
                    throw std::runtime_error("Cannot read SSD optimizer state for saving");
                os.write(reinterpret_cast<const char*>(tmp.data()), pf * sizeof(float));
            } else {
                // 直接从内存池写入 checkpoint
                auto& buf = g_memory_pool.at(i);
                if (!buf.state.empty()) {
                    size_t count = gmeta[i].use ? gmeta[i].proj_elems : ptrs[i]->size();
                    std::memcpy(buf.packed.data(), buf.state[0].data<float>(), count * sizeof(float));
                    std::memcpy(buf.packed.data() + count, buf.state[1].data<float>(), count * sizeof(float));
                    std::memcpy(buf.packed.data() + 2 * count, buf.state[2].data<float>(), ptrs[i]->size() * sizeof(float));
                }
                os.write(reinterpret_cast<const char*>(buf.packed.data()), pf * sizeof(float));
            }
        }
    }

    bool load(std::istream& is) {
        auto read = [&](auto& value) {
            if (!is.read(reinterpret_cast<char*>(&value), sizeof(value)))
                throw std::runtime_error("MLX checkpoint is truncated");
        };
        uint32_t magic = 0;
        read(magic);
        if (magic != CKPT_MAGIC) throw std::runtime_error("Unsupported MLX optimizer checkpoint format");
        int32_t enabled = 0, rank = 0;
        uint64_t min_size = 0;
        read(enabled); read(rank); read(min_size);
        if ((enabled != 0 && enabled != 1) || rank <= 0 || min_size == 0 || min_size > std::numeric_limits<size_t>::max())
            throw std::runtime_error("Invalid GaLore checkpoint configuration");
        size_t n = 0;
        read(n);
        if (n != ptrs.size()) throw std::runtime_error("MLX checkpoint parameter count mismatch");
        ModelParams layout;
        layout.ptrs = ptrs;
        layout.galore_enabled = enabled;
        layout.galore_rank = rank;
        layout.galore_min_size = static_cast<size_t>(min_size);
        layout._build_meta();
        std::vector<array> weights;
        std::vector<std::vector<float>> buffers;
        std::vector<array> finite;
        weights.reserve(n); buffers.reserve(n); finite.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            size_t bytes = 0;
            read(bytes);
            if (bytes != ptrs[i]->nbytes()) throw std::runtime_error("MLX checkpoint parameter shape mismatch");
            auto host = std::make_shared<std::vector<char>>(bytes);
            if (!is.read(host->data(), bytes)) throw std::runtime_error("MLX checkpoint weights are truncated");
            weights.emplace_back(static_cast<void*>(host->data()), ptrs[i]->shape(), ptrs[i]->dtype(), [host](void*) {});
            finite.push_back(all(isfinite(astype(weights.back(), float32))));
            uint64_t count = 0;
            read(count);
            if (count != layout.gmeta[i].packed_floats) throw std::runtime_error("MLX checkpoint optimizer layout mismatch");
            buffers.emplace_back(static_cast<size_t>(count));
            if (!is.read(reinterpret_cast<char*>(buffers.back().data()), count * sizeof(float)))
                throw std::runtime_error("MLX checkpoint optimizer state is truncated");
            for (float value : buffers.back())
                if (!std::isfinite(value)) throw std::runtime_error("MLX checkpoint optimizer contains NaN/Inf");
        }
        if (is.peek() != std::char_traits<char>::eof()) throw std::runtime_error("MLX checkpoint contains unexpected trailing data");
        eval(finite);
        for (const auto& value : finite)
            if (!value.item<bool>()) throw std::runtime_error("MLX checkpoint weights contain NaN/Inf");
        // Commit after validating every tensor; failed restores never mix old/new weights.
        galore_enabled = enabled;
        galore_rank = rank;
        galore_min_size = static_cast<size_t>(min_size);
        set_values(weights);
        init_swap();
        for (size_t i = 0; i < n; ++i) {
            if (use_swap) {
                std::ofstream out(swap_dir + std::to_string(i) + ".bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(buffers[i].data()), buffers[i].size() * sizeof(float));
                out.close();
                if (!out) throw std::runtime_error("Cannot restore SSD optimizer state");
            } else {
                g_memory_pool[i].packed = std::move(buffers[i]);
                g_memory_pool[i].state.clear();
            }
            gmeta[i].p_ready = false;
            gmeta[i].steps_in_subspace = 0;
        }
        return true;
    }

private:
    // 纯内存模式初始化
    void _init_mem() {
        g_memory_pool.clear();
        for (size_t i = 0; i < ptrs.size(); ++i) {
            size_t elements = ptrs[i]->size();
            auto& buf = g_memory_pool[i];
            buf.packed.assign(gmeta[i].packed_floats, 0.0f);

            // master_w 段初始化为当前 FP16 权重的 FP32 镜像。
            // GaLore 参数的 m/v 段是子空间尺寸，master 偏移随之改变，
            // 统一用 _master_offset(i) 取偏移。P 段保持全零，p_ready=false
            // 会让第一次 update 时用当次梯度构建。
            array p_f32 = astype(*ptrs[i], float32);
            eval({p_f32});
            const float* raw = p_f32.data<float>();
            std::memcpy(buf.packed.data() + _master_offset(i), raw, elements * sizeof(float));
        }
    }

    // SSD 交换模式初始化
    void _init_ssd() {
        for (size_t i = 0; i < ptrs.size(); ++i) {
            size_t elements = ptrs[i]->size();
            std::vector<float> packed(gmeta[i].packed_floats, 0.0f);

            array p_f32 = astype(*ptrs[i], float32);
            eval({p_f32});
            const float* raw = p_f32.data<float>();
            std::memcpy(packed.data() + _master_offset(i), raw, elements * sizeof(float));

            std::ofstream os(swap_dir + std::to_string(i) + ".bin", std::ios::binary);
            os.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(float));
        }
    }
};

// 2. SNR-Gated 优化器（自适应双路：纯内存 / SSD 交换）
// packed 布局：[0, N) = m   [N, 2N) = v   [2N, 3N) = master_w

// 单个参数一次更新在 packed 中的落盘信息。优化：不再在每个参数的
// _run_*_update 内部立刻 eval + memcpy（那会让每步发生 74 次 GPU 同步），
// 而是先构建全部参数的更新图，统一 eval 一次、再统一 memcpy 回 packed。
struct ParamOut {
    array m, v, w_master;   // 更新后的优化器状态（FP32，GPU 图节点）
    array param;            // 更新后的模型权重（FP16）
    float* base;            // 该参数 packed 缓冲基址
    size_t m_off, m_cnt;    // m 段偏移与元素数（GaLore 参数为子空间尺寸）
    size_t v_off, v_cnt;
    size_t w_off, w_cnt;

    ParamOut() : m(array(0.0f, float32)), v(array(0.0f, float32)),
                 w_master(array(0.0f, float32)), param(array(0.0f, float32)),
                 base(nullptr), m_off(0), m_cnt(0), v_off(0), v_cnt(0), w_off(0), w_cnt(0) {}
};

static void _run_snr_update(
    array& param, const array& grad, float* packed,
    float lr, float b1_corr, float b2_corr, float max_grad_norm,
    float beta1, float beta2, const array& eps, ParamOut& out,
    TensorBuffer* resident = nullptr)
{
    size_t elements = param.size();

    // Reuse resident arrays; pointer constructors copy only on first use.
    array m = resident && !resident->state.empty() ? resident->state[0] : array(packed, param.shape(), float32);
    array v = resident && !resident->state.empty() ? resident->state[1] : array(packed + elements, param.shape(), float32);
    array w_master = resident && !resident->state.empty() ? resident->state[2] : array(packed + elements * 2, param.shape(), float32);

    array grad_f32 = astype(grad, float32);
    array clipped_grad = grad_f32;
    if (max_grad_norm < 9999.0f) {
        array gnorm = sqrt(sum(square(grad_f32)));
        clipped_grad = where(greater(gnorm, array(max_grad_norm, float32)),
                             multiply(grad_f32, divide(array(max_grad_norm, float32), add(gnorm, eps))),
                             grad_f32);
    }

    m = add(multiply(array(beta1, float32), m), multiply(array(1.0f - beta1, float32), clipped_grad));
    v = add(multiply(array(beta2, float32), v), multiply(array(1.0f - beta2, float32), square(clipped_grad)));

    array m_hat     = divide(m, array(b1_corr, float32));
    array v_hat     = divide(v, array(b2_corr, float32));

    // 优化：sqrt(v_hat)+eps 在 adam_step 和 snr 中被重复计算，提取为公共
    // 图节点 denom——每个参数少建 2 个算子（sqrt+add），数值完全不变。
    array denom     = add(sqrt(v_hat), eps);
    // 优化：adam_step 与 dampening 共享 lr·m_hat。注意 adam_step 必须保持
    // 原乘除顺序 lr·(m_hat/denom)——(m_hat·lr)/denom 会因舍入顺序差 1ulp，
    // 因此只把 lr·m_hat 提取给 dampening 用（取负是精确的符号操作，无舍入）。
    array lr_m      = multiply(m_hat, array(lr, float32));
    array adam_step = multiply(array(lr, float32), divide(m_hat, denom));

    array snr       = divide(abs(m_hat), denom);
    array dampening = multiply(negative(lr_m), array(THRUST_STRENGTH * 0.5f, float32));
    dampening = maximum(minimum(dampening, array(THRUST_MAX * lr, float32)), array(-THRUST_MAX * lr, float32));
    array snr_mask   = greater(snr, array(SNR_THRESHOLD, float32));
    // 优化：where 的第三分支用标量 0 广播替代 zeros_like(dampening)，
    // 每参数少建 1 个算子，语义与数值完全一致。
    array total_step = add(adam_step, where(snr_mask, dampening, array(0.0f, float32)));

    // FP32 霸体高精度迭代，降回 FP16 供前向传播
    w_master = subtract(w_master, total_step);
    param    = astype(w_master, param.dtype());

    // 优化：不在这里 eval+memcpy（会触发一次 GPU 同步）。只记录输出，
    // 由 apply_snr_gated_update 把所有参数的更新图合并成一次 eval。
    out.m = m;
    out.v = v;
    out.w_master = w_master;
    if (resident) resident->state = {m, v, w_master};
    out.param = param;
    out.base = packed;
    out.m_off = 0;               out.m_cnt = elements;
    out.v_off = elements;        out.v_cnt = elements;
    out.w_off = elements * 2;    out.w_cnt = elements;
}

// ---- GaLore 辅助：修正 Gram-Schmidt 列正交化（两遍再正交，数值稳定）----
// Y 是 row-major 的 [n, r] 矩阵。只在子空间重建时调用（每 refresh 步一次），
// 计算量 O(n·r²)：本模型规模下（n≤~16K, r≤256）在 CPU 上毫秒级，不走 GPU
// 是刻意的——避开 MLX linalg API 的版本兼容问题，保持文件零外部依赖。
static void _orthonormalize_columns(std::vector<float>& Y, int n, int r) {
    static std::mt19937 fallback_rng(12345);
    for (int j = 0; j < r; ++j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k < j; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i)
                    dot += (double)Y[(size_t)i * r + j] * Y[(size_t)i * r + k];
                for (int i = 0; i < n; ++i)
                    Y[(size_t)i * r + j] -= (float)(dot * Y[(size_t)i * r + k]);
            }
        }
        double nrm = 0.0;
        for (int i = 0; i < n; ++i)
            nrm += (double)Y[(size_t)i * r + j] * Y[(size_t)i * r + j];
        nrm = std::sqrt(nrm);
        if (nrm < 1e-10) {
            // 该列与前面方向线性相关（梯度秩不足时可能发生）：填随机向量归一。
            // 不再与前列严格正交也没关系——只影响这一列的方向质量，不破坏训练。
            std::normal_distribution<float> nd(0.0f, 1.0f);
            double nn = 0.0;
            for (int i = 0; i < n; ++i) {
                Y[(size_t)i * r + j] = nd(fallback_rng);
                nn += (double)Y[(size_t)i * r + j] * Y[(size_t)i * r + j];
            }
            nn = std::sqrt(std::max(nn, 1e-12));
            for (int i = 0; i < n; ++i) Y[(size_t)i * r + j] = (float)(Y[(size_t)i * r + j] / nn);
        } else {
            for (int i = 0; i < n; ++i) Y[(size_t)i * r + j] = (float)(Y[(size_t)i * r + j] / nrm);
        }
    }
}

// 2b. GaLore 版更新：权重全秩更新，优化器状态存于梯度子空间
// packed 布局：[0, proj) = m_proj  [proj, 2·proj) = v_proj
//              [2·proj, 2·proj+size) = master_w(全尺寸)
//              [2·proj+size, ...) = P 投影矩阵
// 每步流程：G → 投影 Ĝ=PᵀG → 子空间里跑与原版完全相同的 SNR-Gated 逻辑
//           → 投影回 upd=P·step → master_w -= upd
//
// 刷新触发（余弦偏移信号，替代原先的纯固定间隔）：
//   记录构建子空间那一步的梯度 g_ref；之后每步算当前梯度与 g_ref 的余弦
//   相似度，低于 cos_thresh 视为"子空间已经跟不上梯度方向了"，触发重建。
//   配合 min_interval（防止信号噪声导致刷新过密）和 max_interval（兜底：
//   信号长期不触发时的安全上限）。这套机制是针对一个实测问题设计的：
//   固定 refresh=200 时，训练最初200步(损失面变化最剧烈的阶段)几乎全程
//   用同一个子空间硬扛，实测loss从baseline的3.75劣化到5.26；改余弦触发
//   后同等条件下前50步loss从0.157改善到0.092，最终质量还略有提升
//   （Python诊断实验：0.0130 vs baseline 0.0155）。
static void _run_galore_update(
    array& param, const array& grad, float* packed, GaloreMeta& gm,
    float lr, int step, int max_interval, int min_interval, float cos_thresh,
    bool fixed_mode, float max_grad_norm,
    float beta1, float beta2, const array& eps, ParamOut& out,
    TensorBuffer* resident = nullptr,
    const array* prepared_grad = nullptr,
    std::optional<bool> refresh_decision = std::nullopt)
{
    const int rows = gm.rows, cols = gm.cols, r = gm.r;
    const size_t sz   = (size_t)rows * cols;
    const size_t proj = gm.proj_elems;
    float* m_ptr = packed;
    float* v_ptr = packed + proj;
    float* w_ptr = packed + proj * 2;
    float* p_ptr = packed + proj * 2 + sz;

    // 梯度统一 reshape 成 2D 并升 FP32（裁剪逻辑与原版一致）
    array clipped_grad = prepared_grad ? *prepared_grad
        : astype(reshape(grad, {rows, cols}), float32);
    if (!prepared_grad && max_grad_norm < 9999.0f) {
        array gnorm = sqrt(sum(square(clipped_grad)));
        clipped_grad = where(greater(gnorm, array(max_grad_norm, float32)),
                             multiply(clipped_grad, divide(array(max_grad_norm, float32), add(gnorm, eps))),
                             clipped_grad);
    }
    // 注意：这里不提前 eval(clipped_grad)——保持在 MLX 计算图里，让它和后面
    // range finder / SNR更新的算子一起延迟求值，走 GPU/Metal 加速路径。
    // 之前的实现在这里强制 eval 并逐元素手写CPU循环算余弦相似度，导致
    // 每一步都多付一次"图提前落地+CPU标量循环"的开销，是速度从2.5s
    // 退回5s的直接原因。

    // ---- 判定是否需要重建子空间 ----
    int age = step - gm.last_refresh_step;
    bool need_refresh;
    if (refresh_decision.has_value()) {
        need_refresh = *refresh_decision;
    } else if (!gm.p_ready) {
        need_refresh = true;  // 首次构建 / 断点恢复后强制重建
    } else if (age < min_interval) {
        need_refresh = false;  // 安全边界：刚刷新不久，无视信号，防止噪声导致抖动
    } else if (fixed_mode) {
        need_refresh = (age >= max_interval);  // --galore-fixed-refresh：退化为纯固定间隔
    } else if (age >= max_interval) {
        need_refresh = true;  // 兜底：信号长期不触发时的安全上限
    } else {
        // 余弦相似度信号：gm.g_ref（GPU 常驻快照，重建时更新）与当前梯度的
        // 方向偏移。全部用 MLX 算子表达，只在最后 .item<float>() 时同步一个
        // 标量到host。
        //
        // 优化1：g_ref 直接复用持久化 array，不再每步从 host vector 拷贝。
        // 优化2：||g_ref|| 在重建时缓存为 gm.ref_norm，这里只算 n1 一次归约。
        // 优化3：信号检查无需每步都做——min_interval 已保证刷新频率下限，
        //        每 10 步采样一次余弦即可，把 13 个参数 × 每步的同步开销
        //        降到 1/10（min_interval 默认 10，采样更密也不会触发更频繁）。
        if ((age % 10) != 0) {
            need_refresh = false;
        } else {
            array dot = sum(multiply(clipped_grad, gm.g_ref));
            array n1  = sqrt(sum(square(clipped_grad)));
            array denom = add(multiply(n1, array(gm.ref_norm, float32)), array(1e-20f, float32));
            float cos_sim = divide(dot, denom).item<float>();  // 唯一的同步点：一个标量
            need_refresh = (cos_sim < cos_thresh);
        }
    }

    if (need_refresh) {
        // 随机 range finder：Y = G·Ω（投影行侧）或 Gᵀ·Ω（投影列侧），
        // 对 Y 的列做 QR 即得梯度 top-r 奇异子空间的近似正交基。
        // 这是标准的 randomized SVD 第一阶段（Halko et al. 2011），
        // 避免了完整 SVD 的开销和对 MLX linalg 的依赖。
        int big   = gm.project_rows ? rows : cols;
        int small_dim = gm.project_rows ? cols : rows;
        array Omega = random::normal({small_dim, r}, 0.0f, 1.0f);
        array Y = gm.project_rows
                    ? matmul(clipped_grad, astype(Omega, float32))
                    : matmul(transpose(clipped_grad, {1, 0}), astype(Omega, float32));
        eval({Y, clipped_grad});
        std::vector<float> Yh(Y.data<float>(), Y.data<float>() + (size_t)big * r);
        _orthonormalize_columns(Yh, big, r);
        std::memcpy(p_ptr, Yh.data(), (size_t)big * r * sizeof(float));
        // 换了子空间，旧的动量/方差在新坐标系下没有意义，必须清零。
        // Adam 偏置修正会在接下来几步自动补偿冷启动。
        std::memset(m_ptr, 0, proj * sizeof(float));
        std::memset(v_ptr, 0, proj * sizeof(float));
        gm.p_ready = true;
        gm.steps_in_subspace = 0;
        gm.last_refresh_step = step;
        // 记录本次构建时的梯度快照，作为下次判断余弦偏移的参考基准。
        // clipped_grad 此时已经 eval 过（上面那行）。快照直接以 GPU array
        // 形式持久化在 gm.g_ref（拷贝共享底层 buffer，零拷贝），并缓存
        // ||g_ref||，供后续余弦检查复用。
        gm.g_ref = clipped_grad;
        eval({gm.g_ref});
        array ref_norm_arr = sqrt(sum(square(astype(gm.g_ref, float32))));
        eval({ref_norm_arr});
        gm.ref_norm = ref_norm_arr.item<float>();
    }
    gm.steps_in_subspace++;
    // 偏置修正用"子空间内步数"而非全局步数——m/v 在每次重建时清零，
    // 修正系数必须与状态的实际年龄匹配，否则重建后的头几步会严重欠修正
    float b1_corr = 1.0f - std::pow(beta1, (float)gm.steps_in_subspace);
    float b2_corr = 1.0f - std::pow(beta2, (float)gm.steps_in_subspace);

    // 挂载（array 构造会拷贝 host 数据，与原版 _run_snr_update 同一模式）
    const bool cached = resident && !resident->state.empty();
    Shape projected_shape = gm.project_rows ? Shape{r, cols} : Shape{rows, r};
    array m = cached && !need_refresh ? resident->state[0] : array(m_ptr, projected_shape, float32);
    array v = cached && !need_refresh ? resident->state[1] : array(v_ptr, projected_shape, float32);
    array w_master = cached ? resident->state[2] : array(w_ptr, {rows, cols}, float32);
    array P = cached && !need_refresh ? resident->state[3]
        : array(p_ptr, gm.project_rows ? Shape{rows, r} : Shape{cols, r}, float32);

    // 梯度投影进子空间
    array g_hat = gm.project_rows ? matmul(transpose(P, {1, 0}), clipped_grad)
                                  : matmul(clipped_grad, P);

    // ---- 以下与原版 SNR-Gated 逻辑逐行对应，只是发生在子空间坐标里 ----
    m = add(multiply(array(beta1, float32), m), multiply(array(1.0f - beta1, float32), g_hat));
    v = add(multiply(array(beta2, float32), v), multiply(array(1.0f - beta2, float32), square(g_hat)));

    array m_hat     = divide(m, array(b1_corr, float32));
    array v_hat     = divide(v, array(b2_corr, float32));

    // 优化：sqrt(v_hat)+eps 在 adam_step 和 snr 中被重复计算，提取为公共
    // 图节点 denom——每个参数少建 2 个算子（sqrt+add），数值完全不变。
    array denom     = add(sqrt(v_hat), eps);
    // 优化：adam_step 与 dampening 共享 lr·m_hat。注意 adam_step 必须保持
    // 原乘除顺序 lr·(m_hat/denom)——(m_hat·lr)/denom 会因舍入顺序差 1ulp，
    // 因此只把 lr·m_hat 提取给 dampening 用（取负是精确的符号操作，无舍入）。
    array lr_m      = multiply(m_hat, array(lr, float32));
    array adam_step = multiply(array(lr, float32), divide(m_hat, denom));

    array snr       = divide(abs(m_hat), denom);
    array dampening = multiply(negative(lr_m), array(THRUST_STRENGTH * 0.5f, float32));
    dampening = maximum(minimum(dampening, array(THRUST_MAX * lr, float32)), array(-THRUST_MAX * lr, float32));
    array snr_mask   = greater(snr, array(SNR_THRESHOLD, float32));
    // 优化：where 的第三分支用标量 0 广播替代 zeros_like(dampening)，
    // 每参数少建 1 个算子，语义与数值完全一致。
    array total_step = add(adam_step, where(snr_mask, dampening, array(0.0f, float32)));

    // 投影回全秩空间，更新 FP32 master，再降 FP16 供前向
    array upd_full = gm.project_rows ? matmul(P, total_step)
                                     : matmul(total_step, transpose(P, {1, 0}));
    w_master = subtract(w_master, upd_full);
    param    = astype(reshape(w_master, param.shape()), param.dtype());

    // 优化：不在这里 eval+memcpy（会触发一次 GPU 同步）。只记录输出，
    // 由 apply_snr_gated_update 统一 eval 一次后冲刷回 packed。
    // （P 矩阵只在重建时变化，重建时已直接写入 packed，这里不用回写。）
    out.m = m;
    out.v = v;
    out.w_master = w_master;
    out.param = param;
    out.base = packed;
    out.m_off = 0;               out.m_cnt = proj;
    out.v_off = proj;            out.v_cnt = proj;
    out.w_off = proj * 2;        out.w_cnt = sz;
    if (resident) resident->state = {m, v, w_master, P};
}

// 冲刷单个参数：eval 落盘 + memcpy 三段回 packed（SSD 交换路径用，每参数
// 独立 buffer，无法批量）
static void flush_param(ParamOut& out) {
    eval({out.param, out.w_master, out.m, out.v});
    std::memcpy(out.base + out.m_off, out.m.data<float>(),        out.m_cnt * sizeof(float));
    std::memcpy(out.base + out.v_off, out.v.data<float>(),        out.v_cnt * sizeof(float));
    std::memcpy(out.base + out.w_off, out.w_master.data<float>(), out.w_cnt * sizeof(float));
}

void apply_snr_gated_update(ModelParams& mp, const std::vector<array>& grads,
                            float lr, float b1_corr, float b2_corr,
                            float max_grad_norm, int step,
                            const std::vector<array>& metrics = {}) {
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    array eps = array(1e-8f, float32);

    // 优化：先把所有参数的更新图构建好（不 eval），最后统一 eval 一次——
    // 原来每个参数内部 eval+memcpy 会让每步发生 74 次 GPU 同步（profile
    // 显示 apply_snr_gated_update 占主线程 ~43%，其中大量是 Event::wait）。
    // 批量后 GPU 只需同步一次，MLX 还能把 74 个小图合并成一次调度。
    std::vector<ParamOut> pending;
    pending.reserve(mp.ptrs.size());

    // Batch the adaptive GaLore cosine probes. Previously each projected
    // parameter called item<float>() independently, serializing the GPU up to
    // 13 times on every probe step. Materializing all scalar signals together
    // reduces that to one synchronization and reuses the prepared FP32 gradient
    // in the subsequent update graph.
    std::vector<std::optional<array>> prepared_grads(mp.ptrs.size());
    std::vector<std::optional<bool>> refresh_decisions(mp.ptrs.size());
    std::vector<array> cosine_values;
    std::vector<size_t> cosine_indices;
    if (!mp.use_swap && batch_galore_probes) {
        for (size_t i = 0; i < mp.ptrs.size(); ++i) {
            auto& gm = mp.gmeta[i];
            if (!gm.use) continue;
            array clipped = astype(reshape(grads[i], {gm.rows, gm.cols}), float32);
            if (max_grad_norm < 9999.0f) {
                array norm = sqrt(sum(square(clipped)));
                clipped = where(greater(norm, array(max_grad_norm, float32)),
                                multiply(clipped, divide(array(max_grad_norm, float32), add(norm, eps))),
                                clipped);
            }
            prepared_grads[i] = clipped;
            int age = step - gm.last_refresh_step;
            if (!gm.p_ready) refresh_decisions[i] = true;
            else if (age < mp.galore_min_interval) refresh_decisions[i] = false;
            else if (mp.galore_fixed_refresh) refresh_decisions[i] = age >= mp.galore_refresh;
            else if (age >= mp.galore_refresh) refresh_decisions[i] = true;
            else if ((age % 10) != 0) refresh_decisions[i] = false;
            else {
                array dot = sum(multiply(clipped, gm.g_ref));
                array n1 = sqrt(sum(square(clipped)));
                array denom = add(multiply(n1, array(gm.ref_norm, float32)), array(1e-20f, float32));
                cosine_values.push_back(divide(dot, denom));
                cosine_indices.push_back(i);
            }
        }
        if (!cosine_values.empty()) {
            eval(cosine_values);
            for (size_t j = 0; j < cosine_values.size(); ++j)
                refresh_decisions[cosine_indices[j]] =
                    cosine_values[j].item<float>() < mp.galore_cos_thresh;
        }
    }

    for (size_t i = 0; i < mp.ptrs.size(); ++i) {
        array& param      = *(mp.ptrs[i]);
        const array& grad = grads[i];
        GaloreMeta& gm    = mp.gmeta[i];
        const size_t pf   = gm.packed_floats;

        if (mp.use_swap) {
            // ── SSD 交换路径：单次读 → 计算 → 单次写（buffer 是局部的，
            //    无法批量，只能每个参数立即 eval+落盘）──
            std::vector<float> packed(pf);
            {
                std::ifstream is(mp.swap_dir + std::to_string(i) + ".bin", std::ios::binary);
                is.read(reinterpret_cast<char*>(packed.data()), pf * sizeof(float));
            }
            ParamOut out;
            if (gm.use)
                _run_galore_update(param, grad, packed.data(), gm,
                                   lr, step, mp.galore_refresh, mp.galore_min_interval,
                                   mp.galore_cos_thresh, mp.galore_fixed_refresh,
                                   max_grad_norm, beta1, beta2, eps, out);
            else
                _run_snr_update(param, grad, packed.data(),
                                lr, b1_corr, b2_corr, max_grad_norm, beta1, beta2, eps, out);
            flush_param(out);
            {
                std::ofstream os(mp.swap_dir + std::to_string(i) + ".bin", std::ios::binary);
                os.write(reinterpret_cast<const char*>(packed.data()), pf * sizeof(float));
            }
        } else {
            // ── 纯内存路径：直接指针，零额外分配；收集输出待批量 eval ──
            ParamOut out;
            if (gm.use)
                _run_galore_update(param, grad, g_memory_pool.at(i).packed.data(), gm,
                                   lr, step, mp.galore_refresh, mp.galore_min_interval,
                                   mp.galore_cos_thresh, mp.galore_fixed_refresh,
                                   max_grad_norm, beta1, beta2, eps, out,
                                   resident_optimizer ? &g_memory_pool.at(i) : nullptr,
                                   prepared_grads[i] ? &*prepared_grads[i] : nullptr,
                                   refresh_decisions[i]);
            else
                _run_snr_update(param, grad, g_memory_pool.at(i).packed.data(),
                                lr, b1_corr, b2_corr, max_grad_norm, beta1, beta2, eps, out, resident_optimizer ? &g_memory_pool.at(i) : nullptr);
            pending.push_back(std::move(out));
        }
    }

    // Resident outputs survive the step; packed is serialized on save only.
    if (!pending.empty()) {
        std::vector<array> all_outs = metrics;
        all_outs.reserve(pending.size() * 4);
        for (const auto& out : pending) {
            all_outs.push_back(out.param);
            all_outs.push_back(out.w_master);
            all_outs.push_back(out.m);
            all_outs.push_back(out.v);
        }
        eval(all_outs);
        if (!resident_optimizer) {
            for (auto& out : pending) flush_param(out);
        }
    }
    // Cache reuse is bounded by set_cache_limit; explicit clearing is opt-in.
}
// ============================================================================
// 3. 基础神经网络层 
// ============================================================================
struct Linear {
    array W, b;
    Linear(int in, int out, ModelParams& mp) 
        : W(astype(random::normal({in, out}, 0.0f, 0.02f), float16)), 
          b(zeros({out}, float16)) 
    {
        mp.register_param(W);
        mp.register_param(b);
    }
    array operator()(const array& x) const { return add(matmul(x, W), b); }
};

// Official ELF uses Xavier initialization for the self-conditioning and text
// projection path. Keep it local to the ELF adapter so the existing backbone
// initialization remains unchanged.
struct XavierLinear {
    array W, b;
    XavierLinear(int in, int out, ModelParams& mp)
        : W(random::uniform(
              -std::sqrt(6.0f / static_cast<float>(in + out)),
               std::sqrt(6.0f / static_cast<float>(in + out)),
               {in, out}, float16)),
          b(zeros({out}, float16)) {
        mp.register_param(W);
        mp.register_param(b);
    }
    array operator()(const array& x) const { return add(matmul(x, W), b); }
};

struct XavierLinearNoBias {
    array W;
    XavierLinearNoBias(int in, int out, ModelParams& mp)
        : W(random::uniform(
              -std::sqrt(6.0f / static_cast<float>(in + out)),
               std::sqrt(6.0f / static_cast<float>(in + out)),
               {in, out}, float16)) {
        mp.register_param(W);
    }
    array operator()(const array& x) const { return matmul(x, W); }
};

struct QKVLinear {
    int dim;
    array W, b;
    QKVLinear(int d, ModelParams& mp)
        : dim(d),
          W(astype(random::normal({d, 3 * d}, 0.0f, 0.02f), float16)),
          b(zeros({3 * d}, float16)) {
        mp.register_param(W);
        mp.register_param(b);
    }

    std::tuple<array, array, array> operator()(const array& x) const {
        if (fused_qkv_projection) {
            array qkv = add(matmul(x, W), b);
            return {
                slice(qkv, {0, 0}, {x.shape(0), dim}),
                slice(qkv, {0, dim}, {x.shape(0), 2 * dim}),
                slice(qkv, {0, 2 * dim}, {x.shape(0), 3 * dim})};
        }
        array q = add(matmul(x, slice(W, {0, 0}, {dim, dim})),
                      slice(b, {0}, {dim}));
        array k = add(matmul(x, slice(W, {0, dim}, {dim, 2 * dim})),
                      slice(b, {dim}, {2 * dim}));
        array v = add(matmul(x, slice(W, {0, 2 * dim}, {dim, 3 * dim})),
                      slice(b, {2 * dim}, {3 * dim}));
        return {q, k, v};
    }
};

struct RMSNorm {
    float eps;
    array weight;
    RMSNorm(int dim, ModelParams& mp, float e = 1e-4f) 
        : eps(e), weight(ones({dim}, float16)) 
    {
        mp.register_param(weight);
    }
    array operator()(const array& x) const {
        if (fast_rmsnorm) return fast::rms_norm(x, weight, eps);
        array x_f32 = astype(x, float32);
        array var = mean(square(x_f32), -1, true);
        array inv_rms = rsqrt(add(var, array(eps, float32)));
        array out_f32 = multiply(x_f32, inv_rms);
        return multiply(astype(out_f32, float16), weight);
    }
};

struct LoopEmbedding {
    array W;
    LoopEmbedding(int max_loops, int dim, ModelParams& mp) 
        : W(astype(random::normal({max_loops, dim}, 0.0f, 0.02f), float16)) 
    {
        mp.register_param(W);
    }
    array operator()(int t, int batch_size) const {
        array wt = slice(W, {t, 0}, {t+1, W.shape(1)});
        return broadcast_to(wt, {batch_size, W.shape(1)});
    }
};

// ============================================================================
// 4. 注意力机制与 Batched MoE
// ============================================================================
struct AttentionKVCache {
    std::optional<array> k;
    std::optional<array> v;
    int position = 0;
};

static array rope_with_empty_prefix(const array& x, int head_dim) {
    if (rope_empty_prefix_tokens <= 0)
        return fast::rope(x, head_dim, false, 10000.0f, 1.0f, 0);
    if (x.ndim() == 4) {
        int length = x.shape(2);
        if (rope_empty_prefix_tokens >= length) return x;
        array prefix = slice(
            x, {0, 0, 0, 0},
            {x.shape(0), x.shape(1), rope_empty_prefix_tokens, x.shape(3)});
        array text = slice(
            x, {0, 0, rope_empty_prefix_tokens, 0},
            {x.shape(0), x.shape(1), length, x.shape(3)});
        return concatenate(
            {prefix, fast::rope(text, head_dim, false, 10000.0f, 1.0f, 0)}, 2);
    }
    int length = x.shape(1);
    if (rope_empty_prefix_tokens >= length) return x;
    array prefix = slice(
        x, {0, 0, 0},
        {x.shape(0), rope_empty_prefix_tokens, x.shape(2)});
    array text = slice(
        x, {0, rope_empty_prefix_tokens, 0},
        {x.shape(0), length, x.shape(2)});
    return concatenate(
        {prefix, fast::rope(text, head_dim, false, 10000.0f, 1.0f, 0)}, 1);
}

struct SlidingWindowAttention {
    QKVLinear Wqkv;
    Linear Wo;
    int n_heads, head_dim, window_size;

    SlidingWindowAttention(int dim, int h, int ws, ModelParams& mp) 
        : Wqkv(dim, mp), Wo(dim, dim, mp),
          n_heads(h), head_dim(dim/h), window_size(ws) {}

    array operator()(const array& x) const {
        int L = x.shape(0);
        auto [q_raw, k_raw, v_raw] = Wqkv(x);
        if (packed_batch_size > 1) {
            if (L % packed_batch_size != 0)
                throw std::invalid_argument("Packed batch does not divide token count");
            int S = L / packed_batch_size;
            array q = transpose(reshape(q_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            array k = transpose(reshape(k_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            array v = transpose(reshape(v_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            q = rope_with_empty_prefix(q, head_dim);
            k = rope_with_empty_prefix(k, head_dim);
            float scale = 1.0f / std::sqrt((float)head_dim);
            array scores = multiply(matmul(q, transpose(k, {0, 1, 3, 2})),
                                    array(scale, q.dtype()));
            array idx = arange(S);
            array diff = subtract(expand_dims(idx, 1), expand_dims(idx, 0));
            array mask = bidirectional_training_attention
                ? less(abs(diff), array(window_size))
                : logical_and(greater_equal(diff, array(0)),
                              less(diff, array(window_size)));
            if (rope_empty_prefix_tokens > 0) {
                array condition_link = logical_or(
                    less(expand_dims(idx, 1), array(rope_empty_prefix_tokens)),
                    less(expand_dims(idx, 0), array(rope_empty_prefix_tokens)));
                mask = logical_or(mask, condition_link);
            }
            scores = where(mask, scores, array(-1e4f, scores.dtype()));
            array probs = astype(softmax(astype(scores, float32), -1), float16);
            array out = matmul(probs, v);
            out = reshape(transpose(out, {0, 2, 1, 3}),
                          {L, n_heads * head_dim});
            return Wo(out);
        }
        array q = q_raw; array k = k_raw; array v = v_raw;
        
        q = transpose(reshape(q, {L, n_heads, head_dim}), {1, 0, 2});
        k = transpose(reshape(k, {L, n_heads, head_dim}), {1, 0, 2});
        v = transpose(reshape(v, {L, n_heads, head_dim}), {1, 0, 2});

        q = rope_with_empty_prefix(q, head_dim);
        k = rope_with_empty_prefix(k, head_dim);

        float scale = 1.0f / std::sqrt((float)head_dim);
        array scores = multiply(matmul(q, transpose(k, {0, 2, 1})), array(scale, q.dtype()));

        array idx = arange(L);
        array diff = subtract(expand_dims(idx, 1), expand_dims(idx, 0));
        array mask = bidirectional_training_attention
            ? less(abs(diff), array(window_size))
            : logical_and(greater_equal(diff, array(0)),
                          less(diff, array(window_size)));
        if (rope_empty_prefix_tokens > 0) {
            array condition_link = logical_or(
                less(expand_dims(idx, 1), array(rope_empty_prefix_tokens)),
                less(expand_dims(idx, 0), array(rope_empty_prefix_tokens)));
            mask = logical_or(mask, condition_link);
        }
        
        scores = where(mask, scores, array(-1e4f, scores.dtype()));
        array probs = astype(softmax(astype(scores, float32), -1), float16);
        array out = matmul(probs, v);
        
        out = reshape(transpose(out, {1, 0, 2}), {L, n_heads * head_dim});
        return Wo(out);
    }

    array decode(const array& x, AttentionKVCache& cache) const {
        auto [q_raw, k_raw, v_raw] = Wqkv(x);
        array q = transpose(reshape(q_raw, {1, n_heads, head_dim}), {1, 0, 2});
        array k = transpose(reshape(k_raw, {1, n_heads, head_dim}), {1, 0, 2});
        array v = transpose(reshape(v_raw, {1, n_heads, head_dim}), {1, 0, 2});
        q = fast::rope(q, head_dim, false, 10000.0f, 1.0f, cache.position);
        k = fast::rope(k, head_dim, false, 10000.0f, 1.0f, cache.position);

        array keys = cache.k ? concatenate({*cache.k, k}, 1) : k;
        array values = cache.v ? concatenate({*cache.v, v}, 1) : v;
        int length = keys.shape(1);
        if (length > window_size) {
            keys = slice(keys, {0, length - window_size, 0},
                         {n_heads, length, head_dim});
            values = slice(values, {0, length - window_size, 0},
                           {n_heads, length, head_dim});
            length = window_size;
        }
        cache.k = keys;
        cache.v = values;
        cache.position++;

        float scale = 1.0f / std::sqrt((float)head_dim);
        array out = array(0.0f, float16);
        if (fast_decode_sdpa) {
            // q_len=1 and head_dim=96 in the production model hit MLX's
            // vector SDPA Metal kernel. Keep cache storage rank-3 to avoid
            // changing the surrounding decode graph, and add batch only at
            // the fused primitive boundary.
            out = fast::scaled_dot_product_attention(
                expand_dims(q, 0), expand_dims(keys, 0),
                expand_dims(values, 0), scale);
            out = reshape(transpose(out, {0, 2, 1, 3}),
                          {1, n_heads * head_dim});
        } else {
            array scores = multiply(matmul(q, transpose(keys, {0, 2, 1})),
                                    array(scale, q.dtype()));
            array probs = astype(softmax(astype(scores, float32), -1), float16);
            out = matmul(probs, values);
            out = reshape(transpose(out, {1, 0, 2}),
                          {1, n_heads * head_dim});
        }
        return Wo(out);
    }
};

struct SparseGlobalAttention {
    QKVLinear Wqkv;
    Linear Wo;
    int n_heads, head_dim, topk_n;

    SparseGlobalAttention(int dim, int h, int tk, ModelParams& mp)
        : Wqkv(dim, mp), Wo(dim, dim, mp),
          n_heads(h), head_dim(dim/h), topk_n(tk) {}

    array operator()(const array& x) const {
        int L = x.shape(0);
        auto [q_raw, k_raw, v_raw] = Wqkv(x);
        if (packed_batch_size > 1) {
            if (L % packed_batch_size != 0)
                throw std::invalid_argument("Packed batch does not divide token count");
            int S = L / packed_batch_size;
            array q = transpose(reshape(q_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            array k = transpose(reshape(k_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            array v = transpose(reshape(v_raw, {packed_batch_size, S, n_heads, head_dim}),
                                {0, 2, 1, 3});
            q = rope_with_empty_prefix(q, head_dim);
            k = rope_with_empty_prefix(k, head_dim);
            float scale = 1.0f / std::sqrt((float)head_dim);
            array scores = multiply(matmul(q, transpose(k, {0, 1, 3, 2})),
                                    array(scale, q.dtype()));
            array idx = arange(S);
            if (!bidirectional_training_attention) {
                array mask = greater_equal(expand_dims(idx, 1), expand_dims(idx, 0));
                scores = where(mask, scores, array(-1e4f, scores.dtype()));
            }
            int k_actual = std::min(topk_n, S);
            array threshold = min(topk(scores, k_actual, -1), -1, true);
            array keep = greater_equal(scores, threshold);
            if (rope_empty_prefix_tokens > 0) {
                array condition_keys = reshape(
                    less(idx, array(rope_empty_prefix_tokens)), {1, 1, 1, S});
                keep = logical_or(keep, condition_keys);
            }
            scores = where(keep, scores,
                           array(-1e4f, scores.dtype()));
            array probs = astype(softmax(astype(scores, float32), -1), float16);
            array out = matmul(probs, v);
            out = reshape(transpose(out, {0, 2, 1, 3}),
                          {L, n_heads * head_dim});
            return Wo(out);
        }
        array q = q_raw; array k = k_raw; array v = v_raw;
        
        q = transpose(reshape(q, {L, n_heads, head_dim}), {1, 0, 2});
        k = transpose(reshape(k, {L, n_heads, head_dim}), {1, 0, 2});
        v = transpose(reshape(v, {L, n_heads, head_dim}), {1, 0, 2});

        q = rope_with_empty_prefix(q, head_dim);
        k = rope_with_empty_prefix(k, head_dim);

        float scale = 1.0f / std::sqrt((float)head_dim);
        array scores = multiply(matmul(q, transpose(k, {0, 2, 1})), array(scale, q.dtype()));

        array idx = arange(L);
        if (!bidirectional_training_attention) {
            array mask_c = greater_equal(expand_dims(idx, 1), expand_dims(idx, 0));
            scores = where(mask_c, scores, array(-1e4f, scores.dtype()));
        }

        int k_actual = std::min(topk_n, L);
        if (gather_sparse_attention) {
            array selected = stop_gradient(astype(slice(
                argpartition(scores, L - k_actual, -1),
                {0, 0, L - k_actual}, {n_heads, L, L}), int32));
            array selected_scores = take_along_axis(scores, selected, -1);
            array selected_probs = astype(
                softmax(astype(selected_scores, float32), -1), float16);
            std::vector<array> head_outputs;
            head_outputs.reserve(n_heads);
            for (int h = 0; h < n_heads; ++h) {
                array idx_h = reshape(slice(selected, {h, 0, 0},
                                            {h + 1, L, k_actual}), {L * k_actual});
                array value_h = reshape(slice(v, {h, 0, 0},
                                              {h + 1, L, head_dim}), {L, head_dim});
                array gathered = reshape(take(value_h, idx_h, 0),
                                         {L, k_actual, head_dim});
                array prob_h = reshape(slice(selected_probs, {h, 0, 0},
                                             {h + 1, L, k_actual}), {L, k_actual, 1});
                head_outputs.push_back(sum(multiply(gathered, prob_h), 1));
            }
            array out = reshape(transpose(stack(head_outputs, 0), {1, 0, 2}),
                                {L, n_heads * head_dim});
            return Wo(astype(out, float16));
        }
        array topk_vals = topk(scores, k_actual, -1);
        // MLX topk returns an unsorted partition. The last element is not the
        // kth-largest threshold; reduce over the returned set instead.
        array threshold = min(topk_vals, -1, true);
        array keep = greater_equal(scores, threshold);
        if (rope_empty_prefix_tokens > 0) {
            array condition_keys = reshape(
                less(idx, array(rope_empty_prefix_tokens)), {1, 1, L});
            keep = logical_or(keep, condition_keys);
        }
        scores = where(keep, scores, array(-1e4f, scores.dtype()));

        array probs = astype(softmax(astype(scores, float32), -1), float16);
        array out = matmul(probs, v);
        out = reshape(transpose(out, {1, 0, 2}), {L, n_heads * head_dim});
        return Wo(out);
    }

    array decode(const array& x, AttentionKVCache& cache, int max_context) const {
        auto [q_raw, k_raw, v_raw] = Wqkv(x);
        array q = transpose(reshape(q_raw, {1, n_heads, head_dim}), {1, 0, 2});
        array k = transpose(reshape(k_raw, {1, n_heads, head_dim}), {1, 0, 2});
        array v = transpose(reshape(v_raw, {1, n_heads, head_dim}), {1, 0, 2});
        q = fast::rope(q, head_dim, false, 10000.0f, 1.0f, cache.position);
        k = fast::rope(k, head_dim, false, 10000.0f, 1.0f, cache.position);

        array keys = cache.k ? concatenate({*cache.k, k}, 1) : k;
        array values = cache.v ? concatenate({*cache.v, v}, 1) : v;
        int length = keys.shape(1);
        if (length > max_context) {
            keys = slice(keys, {0, length - max_context, 0},
                         {n_heads, length, head_dim});
            values = slice(values, {0, length - max_context, 0},
                           {n_heads, length, head_dim});
            length = max_context;
        }
        cache.k = keys;
        cache.v = values;
        cache.position++;

        float scale = 1.0f / std::sqrt((float)head_dim);
        array scores = multiply(matmul(q, transpose(keys, {0, 2, 1})),
                                array(scale, q.dtype()));
        int k_actual = std::min(topk_n, length);
        array selected = argpartition(scores, length - k_actual, -1);
        selected = astype(slice(selected, {0, 0, length - k_actual},
                                {n_heads, 1, length}), int32);
        array selected_scores = take_along_axis(scores, selected, -1);
        array probs = astype(softmax(astype(selected_scores, float32), -1), float16);

        array out = array(0.0f, float16);
        if (vectorized_sparse_decode) {
            // Gather [head, top_k, head_dim] in one graph node family instead
            // of building n_heads independent slice/take/multiply reductions.
            array gather_idx = broadcast_to(
                transpose(selected, {0, 2, 1}),
                {n_heads, k_actual, head_dim});
            array selected_values = take_along_axis(values, gather_idx, 1);
            array weights = transpose(probs, {0, 2, 1});
            out = reshape(sum(multiply(selected_values, weights), 1),
                          {1, n_heads * head_dim});
        } else {
            std::vector<array> head_outputs;
            head_outputs.reserve(n_heads);
            for (int h = 0; h < n_heads; ++h) {
                array idx_h = reshape(slice(selected, {h, 0, 0},
                                            {h + 1, 1, k_actual}), {k_actual});
                array value_h = reshape(slice(values, {h, 0, 0},
                                              {h + 1, length, head_dim}),
                                        {length, head_dim});
                array prob_h = reshape(slice(probs, {h, 0, 0},
                                             {h + 1, 1, k_actual}), {k_actual, 1});
                head_outputs.push_back(sum(
                    multiply(take(value_h, idx_h, 0), prob_h), 0));
            }
            out = reshape(stack(head_outputs, 0), {1, n_heads * head_dim});
        }
        return Wo(astype(out, float16));
    }
};

struct BatchedMoE {
    int dim, n_experts, top_k;
    Linear router;
    array W_gate, W_up, W_down; 

    BatchedMoE(int d, int h, int ne, int tk, ModelParams& mp) 
        : dim(d), n_experts(ne), top_k(tk), router(d, ne, mp),
          W_gate(astype(random::normal({ne, d, h}, 0.0f, 0.02f), float16)),
          W_up(astype(random::normal({ne, d, h}, 0.0f, 0.02f), float16)),
          W_down(astype(random::normal({ne, h, d}, 0.0f, 0.02f), float16)) {
        mp.register_param(W_gate);
        mp.register_param(W_up);
        mp.register_param(W_down);
    }

    array operator()(const array& x) const {
        int L = x.shape(0);
        array l = router(x);
        array probs = astype(softmax(astype(l, float32), -1), float16);

        array topk_vals = topk(probs, top_k, -1);
        array threshold = slice(topk_vals, {0, top_k - 1}, {L, top_k});
        array route_probs = where(greater_equal(probs, threshold), probs, zeros_like(probs));
        route_probs = divide(route_probs, expand_dims(sum(route_probs, -1), -1)); 

        array x_expanded = broadcast_to(expand_dims(x, 0), {n_experts, L, dim});

        array g = sigmoid(matmul(x_expanded, W_gate));
        array up = matmul(x_expanded, W_up);          
        array hidden = multiply(g, up);               
        array out_experts = matmul(hidden, W_down);    

        array w = expand_dims(transpose(route_probs, {1, 0}), -1);
        array weighted_out = multiply(out_experts, w); 
        
        return sum(weighted_out, 0); 
    }
};

struct ACT {
    int max_loops;
    Linear linear;
    ACT(int d, int l, ModelParams& mp) : max_loops(l), linear(d, 1, mp) {}

    array operator()(const std::vector<array>& hs) const {
        int L = hs[0].shape(0);
        array out = zeros_like(hs[0]);
        array rem = ones({L, 1}, float16);

        for (size_t t = 0; t < hs.size(); ++t) {
            array pt = sigmoid(linear(hs[t]));
            array wt = (t == hs.size() - 1) ? rem : multiply(pt, rem);
            rem = subtract(rem, wt);
            out = add(out, multiply(hs[t], wt));
        }
        return out;
    }
};

// ============================================================================
// 4.5 Capacity-based 稀疏 MoE（新增，可选替代 BatchedMoE）
// ============================================================================
// 详细设计依据见独立验证文件 capacity_moe.h / test_capacity_moe.cpp。
// 已在 test_capacity_moe.cpp 中通过：前向数值健全性、专家分配均衡性、
// 反向传播梯度（用 stop_gradient 修复过 "[scatter] Cannot calculate VJP
// with respect to indices" 问题）三项独立测试。
//
// 集成方式：保留原 BatchedMoE 不删，下方 USE_CAPACITY_MOE 宏控制
// TransformerBlock/RecurrentBlock 实际使用哪一个，方便随时切换对比。
struct CapacityMoE {
    int dim, n_experts, top_k, capacity;
    float capacity_factor;
    Linear router;
    array W_gate, W_up, W_down;

    // 路由噪声强度：训练时给router的logits加一点高斯噪声，目的是打破
    // "赢家通吃"的专家坍缩循环——如果路由器对某几个专家有持续的微弱偏好，
    // 不加干预这个偏好会被梯度自我强化；加噪声后，偶尔有token会被"意外"
    // 分配给本不占优的专家，让那些专家也能获得训练信号、变得有竞争力。
    // 这是 Noisy Top-K Gating 的标准做法（Shazeer et al. 2017），和负载
    // 均衡辅助损失是互补关系，不是替代——前者靠"强制探索"打破固化，
    // 后者靠"惩罚不均衡"引导优化方向，两者一起用通常比单独用任一个更有效。
    //
    // is_training 控制噪声是否生效：训练时希望路由有一定随机探索性，但
    // 推理/生成阶段应该用模型学到的、确定性的最优路由判断，不应该再随机
    // 扰动——否则同样的输入可能产生不一致的输出，且没有探索的必要性
    // （推理时没有梯度更新，"探索"不会带来任何好处，只会增加结果的方差）。
    mutable bool is_training = true;
    float router_noise_std;

    CapacityMoE(int d, int h, int ne, int tk, int seq_len, ModelParams& mp,
                float capacity_factor = 1.4f,  // 根据实测随机负载分布，从1.25上调到1.4留更多余量
                float noise_std = 0.0f)
        : dim(d), n_experts(ne), top_k(tk), capacity_factor(capacity_factor),
          router(d, ne, mp),
          W_gate(astype(random::normal({ne, d, h}, 0.0f, 0.02f), float16)),
          W_up(astype(random::normal({ne, d, h}, 0.0f, 0.02f), float16)),
          W_down(astype(random::normal({ne, h, d}, 0.0f, 0.02f), float16)),
          router_noise_std(noise_std)
    {
        capacity = static_cast<int>(std::ceil(
            (float)seq_len * tk / ne * capacity_factor));
        mp.register_param(W_gate);
        mp.register_param(W_up);
        mp.register_param(W_down);
    }

    std::pair<array, array> forward_with_aux(const array& x) const {
        int L = x.shape(0);
        int route_batches = packed_batch_size > 1 ? packed_batch_size : 1;
        int route_sequence_length = L / route_batches;
        int route_capacity = (!is_training && incremental_decode_active)
            ? std::max(1, std::min(capacity, static_cast<int>(std::ceil(
                  (float)L * top_k / n_experts * capacity_factor))))
            : capacity;

        array l = router(x);
        array probs = astype(softmax(astype(l, float32), -1), float32);

        // argsort/cumsum/equal 等操作产出的都是"用于决策路由去向"的整数索引，
        // 不应该携带梯度——它们最终会被传给 scatter 的 indices 参数，而 MLX
        // 不支持对 scatter 的 indices 求导。这里在 probs 分流出"决策路径"的
        // 地方立刻 stop_gradient，避免梯度图里出现非法依赖。
        array probs_for_routing = stop_gradient(probs);

        // 路由噪声只加在这条"用于决策选哪个专家"的路径上，不污染下面
        // top_val（真正决定输出权重大小、带梯度参与训练的那部分）。
        // 这样噪声只影响"选择"这个离散行为本身，不会让梯度信号本身失真。
        if (is_training && router_noise_std > 0.0f) {
            array noise = random::normal(probs_for_routing.shape(), 0.0f, router_noise_std);
            probs_for_routing = add(probs_for_routing, noise);
        }

        array sorted_idx = partition_moe_topk
            ? argpartition(probs_for_routing, n_experts - top_k, -1)
            : argsort(probs_for_routing, -1);
        array top_idx = slice(sorted_idx, {0, n_experts - top_k}, {L, n_experts});
        top_idx = astype(top_idx, int32);

        // top_val 仍从原始（带梯度、不含噪声）的 probs 取值，路由权重的
        // 梯度链路完整保留，不受噪声干扰
        array top_val = take_along_axis(probs, top_idx, -1);
        array top_val_sum = sum(top_val, -1, true);
        top_val = divide(top_val, top_val_sum);

        int N = L * top_k;
        array flat_token = reshape(
            broadcast_to(expand_dims(arange(L), 1), {L, top_k}), {N});
        array flat_expert = reshape(top_idx, {N});
        array flat_weight = reshape(top_val, {N});  // 带梯度，决定输出贡献大小

        array expert_range = arange(n_experts);
        array one_hot = astype(
            equal(expand_dims(flat_expert, 1), expand_dims(expert_range, 0)),
            float32);
        array cum_count = zeros_like(one_hot);
        array flat_batch = zeros({N}, int32);
        if (route_batches > 1) {
            array grouped_one_hot = reshape(
                one_hot, {route_batches, route_sequence_length * top_k, n_experts});
            cum_count = reshape(
                subtract(cumsum(grouped_one_hot, 1), array(1.0f, float32)),
                {N, n_experts});
            array token_batch = reshape(broadcast_to(
                expand_dims(arange(route_batches), 1),
                {route_batches, route_sequence_length}), {L});
            flat_batch = reshape(broadcast_to(
                expand_dims(token_batch, 1), {L, top_k}), {N});
        } else {
            cum_count = subtract(cumsum(one_hot, 0), array(1.0f, float32));
        }
        array slot_per_item = sum(multiply(cum_count, one_hot), -1);
        array slot_int = astype(slot_per_item, int32);

        array keep_mask = less(slot_int, array(route_capacity, int32));
        array clipped_slot = minimum(slot_int, array(route_capacity - 1, int32));
        array batch_expert = add(
            multiply(flat_batch, array(n_experts, int32)), flat_expert);
        array dest = add(multiply(batch_expert, array(route_capacity, int32)),
                         clipped_slot);
        int trash_idx = route_batches * n_experts * route_capacity;
        dest = where(keep_mask, dest, array(trash_idx, int32));
        // 双重保险：确保 dest 以纯数值常量身份进入 scatter
        dest = stop_gradient(dest);

        // safe_weight 从 flat_weight（带梯度）经过 where 筛选，梯度链路要保留
        array safe_weight = where(keep_mask, flat_weight, array(0.0f, float32));

        int buf_size = route_batches * n_experts * route_capacity + 1;
        array x_f32 = astype(x, float32);
        array gathered_x = take(x_f32, flat_token, 0);

        array expert_input_buf = zeros({buf_size, dim}, float32);
        expert_input_buf = scatter(expert_input_buf, {dest}, expand_dims(gathered_x, 1), std::vector<int>{0});

        array expert_weight_buf = zeros({buf_size}, float32);
        expert_weight_buf = scatter(expert_weight_buf, {dest},
                                     expand_dims(safe_weight, -1), std::vector<int>{0});

        // token_id_buf 纯粹是"记录写回位置"的簿记信息，不参与数值计算
        array token_id_buf = full({buf_size}, array(-1), int32);
        token_id_buf = scatter(token_id_buf, {dest},
                                expand_dims(flat_token, -1), std::vector<int>{0});

        array expert_input = slice(
            expert_input_buf, {0, 0},
            {route_batches * n_experts * route_capacity, dim});
        expert_input = route_batches > 1
            ? reshape(expert_input,
                      {route_batches, n_experts, route_capacity, dim})
            : reshape(expert_input, {n_experts, route_capacity, dim});
        expert_input = astype(expert_input, float16);

        array g = sigmoid(matmul(expert_input, astype(W_gate, float16)));
        array up = matmul(expert_input, astype(W_up, float16));
        array hidden_act = multiply(g, up);
        array out_experts = matmul(hidden_act, astype(W_down, float16));

        array weight_buf = slice(
            expert_weight_buf, {0},
            {route_batches * n_experts * route_capacity});
        weight_buf = route_batches > 1
            ? reshape(weight_buf,
                      {route_batches, n_experts, route_capacity, 1})
            : reshape(weight_buf, {n_experts, route_capacity, 1});
        array weighted = multiply(astype(out_experts, float32),
                                   astype(weight_buf, float32));

        array flat_out = reshape(
            weighted, {route_batches * n_experts * route_capacity, dim});
        array flat_tok_id = slice(
            token_id_buf, {0}, {route_batches * n_experts * route_capacity});

        array out = zeros({L, dim}, float32);
        array valid_mask = greater_equal(flat_tok_id, array(0, int32));
        // safe_tok_id 用作 scatter_add 的 indices，必须不带梯度
        array safe_tok_id = stop_gradient(where(valid_mask, flat_tok_id, array(0, int32)));
        array masked_out = multiply(flat_out,
                                     astype(expand_dims(valid_mask, -1), float32));

        out = scatter_add(out, {safe_tok_id}, expand_dims(masked_out, 1), std::vector<int>{0});

        // Reuse the exact router probabilities and dispatch decisions above for
        // the Switch-style balancing loss. This removes a second
        // router/softmax/argsort/one-hot subgraph for every MoE invocation.
        array aux = array(0.0f, float32);
        if (packed_batch_size > 1) {
            int sequence_length = L / packed_batch_size;
            array P = mean(reshape(probs,
                                   {packed_batch_size, sequence_length, n_experts}),
                           1);
            array dispatch_count = sum(reshape(
                one_hot, {packed_batch_size, sequence_length * top_k, n_experts}), 1);
            array f = stop_gradient(divide(
                dispatch_count, array((float)(sequence_length * top_k), float32)));
            aux = mean(multiply(array((float)n_experts, float32),
                                sum(multiply(f, P), -1)));
        } else {
            array P = mean(probs, 0);
            array dispatch_count = sum(one_hot, 0);
            array f = stop_gradient(divide(
                dispatch_count, array((float)(L * top_k), float32)));
            aux = multiply(array((float)n_experts, float32), sum(multiply(f, P)));
        }
        return {astype(out, float16), aux};
    }

    array operator()(const array& x) const { return forward_with_aux(x).first; }

    // 调试辅助：返回每个专家实际分配的token数，便于训练时监控是否频繁溢出
    array debug_fill_count(const array& x) const {
        int L = x.shape(0);
        array l = router(x);
        array probs = astype(softmax(astype(l, float32), -1), float32);
        array sorted_idx = argsort(probs, -1);
        array top_idx = astype(slice(sorted_idx, {0, n_experts - top_k}, {L, n_experts}), int32);
        array flat_expert = reshape(top_idx, {L * top_k});
        array expert_range = arange(n_experts);
        array one_hot = astype(
            equal(expand_dims(flat_expert, 1), expand_dims(expert_range, 0)), float32);
        if (packed_batch_size > 1) {
            int sequence_length = L / packed_batch_size;
            return sum(reshape(
                one_hot,
                {packed_batch_size, sequence_length * top_k, n_experts}), 1);
        }
        return sum(one_hot, 0);  // [n_experts] 每个专家命中的token总数（未截断前）
    }

    // 负载均衡辅助损失（Switch Transformer 风格），用于抑制"专家坍缩"——
    // 路由器一旦偶然偏好某几个专家，那几个专家就能拿到更多梯度训练得更好，
    // 变得更"有吸引力"，进一步被更多token选中，形成自我强化的恶性循环。
    //
    // 公式：aux_loss = n_experts * Σ_e (f_e * P_e)
    //   f_e = 实际分配给专家e的token比例（离散统计量，stop_gradient，
    //         因为"有多少token选了这个专家"本身不是连续可导的量）
    //   P_e = 专家e在全部token上的平均路由概率（来自softmax，连续可导，
    //         这是优化器唯一能通过梯度去调整的部分）
    // 这个损失的设计巧妙之处：只有当 f_e（实际分配多）和 P_e（平均概率高）
    // 同时偏高时，损失才会大——这会推动优化器把概率质量从"已经很受欢迎"的
    // 专家身上挪开，而不需要对f_e本身求导（它原本也没法求导）。
    //
    // 注意：这里重新跑了一次 router(x) + softmax，是对 operator() 内部计算
    // 的小幅重复（router本身只是一个 Linear+softmax，相对于后面的FFN专家
    // 计算量很小，重复这一小部分的代价可以接受），这样做是为了不改动
    // operator() 的返回签名，避免牵连 TransformerBlock/RecurrentBlock 等
    // 调用点。
    array load_balance_loss(const array& x) const {
        int L = x.shape(0);
        array l = router(x);
        array probs = astype(softmax(astype(l, float32), -1), float32);  // [L, n_experts] 带梯度

        // P_e：每个专家在全部token上的平均路由概率（带梯度，这是aux_loss
        // 真正能影响参数的部分）
        array P = mean(probs, 0);  // [n_experts]

        // f_e：每个专家实际拿到的top-k分配比例（离散，不带梯度）
        array probs_for_routing = stop_gradient(probs);
        array sorted_idx = argsort(probs_for_routing, -1);
        array top_idx = astype(slice(sorted_idx, {0, n_experts - top_k}, {L, n_experts}), int32);
        array flat_expert = reshape(top_idx, {L * top_k});
        array expert_range = arange(n_experts);
        array one_hot = astype(
            equal(expand_dims(flat_expert, 1), expand_dims(expert_range, 0)), float32);
        array dispatch_count = sum(one_hot, 0);  // [n_experts]
        array f = stop_gradient(divide(dispatch_count, array((float)(L * top_k), float32)));

        array aux = multiply(array((float)n_experts, float32), sum(multiply(f, P)));
        return aux;
    }
};

// 切换开关：定义此宏则用 CapacityMoE，否则保持原 BatchedMoE
// Build option: TURTLE_CAPACITY_MOE in mlx/CMakeLists.txt.

// ============================================================================
// 5. 模型块装配
// ============================================================================
struct TransformerBlock {
    SlidingWindowAttention attn_sw;
#ifdef USE_CAPACITY_MOE
    CapacityMoE moe;
#else
    BatchedMoE moe;
#endif
    RMSNorm n_attn, n_moe;

    // side-channel：在 operator() 这个 const 方法内部记录本次forward产生的
    // 负载均衡辅助损失。用 mutable 是因为 MLX 的 array 本身是值类型+惰性求值，
    // 这里只是存一个计算图节点的引用，不是真正"修改状态"，训练循环读取后
    // 累加进总loss参与反向传播即可。
    mutable array aux_loss = array(0.0f, float32);

    TransformerBlock(int dim, int ws, int ne, int tk, int seq_len, ModelParams& mp,
                      float capacity_factor = 1.4f, float noise_std = 0.0f)
        : attn_sw(dim, 8, ws, mp),
#ifdef USE_CAPACITY_MOE
          moe(dim, dim * 4, ne, tk, seq_len, mp, capacity_factor, noise_std),
#else
          moe(dim, dim * 4, ne, tk, mp),
#endif
          n_attn(dim, mp), n_moe(dim, mp) {}

    array operator()(const array& x) const {
        array mid = add(x, attn_sw(n_attn(x)));
        array moe_in = n_moe(mid);
#ifdef USE_CAPACITY_MOE
        if (fused_moe_aux) {
            auto moe_result = moe.forward_with_aux(moe_in);
            aux_loss = moe_result.second;
            return add(mid, moe_result.first);
        }
        aux_loss = moe.load_balance_loss(moe_in);
#endif
        return add(mid, moe(moe_in));
    }

    array decode(const array& x, AttentionKVCache& cache) const {
        array mid = add(x, attn_sw.decode(n_attn(x), cache));
        return add(mid, moe(n_moe(mid)));
    }
};

struct RecurrentKVCache {
    std::vector<AttentionKVCache> wide;
    std::vector<AttentionKVCache> loops;
    explicit RecurrentKVCache(size_t wide_count = 0, size_t loop_count = 0)
        : wide(wide_count), loops(loop_count) {}
};

struct RecurrentBlock {
    std::vector<std::shared_ptr<TransformerBlock>> wide_blocks;
    LoopEmbedding loop_embed;
    SlidingWindowAttention attn_sw;
    SparseGlobalAttention attn_global;
#ifdef USE_CAPACITY_MOE
    CapacityMoE moe;
#else
    BatchedMoE moe;
#endif
    RMSNorm n_attn, n_moe;
    ACT act;
    float memory_alpha = 0.9f;

    // 同 TransformerBlock：side-channel 记录本次forward的负载均衡辅助损失。
    // 注意 RecurrentBlock 内部的 moe 在 ACT 循环里被调用 max_loop 次，
    // 所以这里要在循环开始前清零，循环内逐次累加。
    mutable array aux_loss = array(0.0f, float32);

    RecurrentBlock(int dim, int max_loops, int ws, int gk, int ne, int tk, int nw, int seq_len, ModelParams& mp,
                    float capacity_factor = 1.4f, float noise_std = 0.0f)
        : loop_embed(max_loops, dim, mp), attn_sw(dim, 8, ws, mp), attn_global(dim, 8, gk, mp),
#ifdef USE_CAPACITY_MOE
          moe(dim, dim * 4, ne, tk, seq_len, mp, capacity_factor, noise_std),
#else
          moe(dim, dim * 4, ne, tk, mp),
#endif
          n_attn(dim, mp), n_moe(dim, mp), act(dim, max_loops, mp) {
        for (int i = 0; i < nw; ++i) {
            wide_blocks.push_back(std::make_shared<TransformerBlock>(dim, ws, ne, tk, seq_len, mp, capacity_factor, noise_std));
        }
    }

    array operator()(array x) const {
        for (auto& blk : wide_blocks) x = blk->operator()(x);
        
        std::vector<array> hs;
        array memory = zeros_like(x);
#ifdef USE_CAPACITY_MOE
        aux_loss = array(0.0f, float32);  // 每次forward开始前清零，避免跨step累积
#endif

        for (int t = 0; t < act.max_loops; ++t) {
            array n_a = add(n_attn(x), multiply(memory, array(0.1f, float16)));
            x = add(x, loop_embed(t, x.shape(0)));
            array a_out = (t % 2 == 0) ? attn_sw(n_a) : attn_global(n_a);
            
            array mid = add(x, a_out);
            array moe_in = n_moe(mid);
#ifdef USE_CAPACITY_MOE
            if (fused_moe_aux) {
                auto moe_result = moe.forward_with_aux(moe_in);
                aux_loss = add(aux_loss, moe_result.second);
                x = add(mid, moe_result.first);
            } else {
                aux_loss = add(aux_loss, moe.load_balance_loss(moe_in));
                x = add(mid, moe(moe_in));
            }
#else
            x = add(mid, moe(moe_in));
#endif
            
            memory = add(multiply(array(memory_alpha, float16), memory), multiply(array(1.0f - memory_alpha, float16), x));
            hs.push_back(x);
        }
        return act(hs);
    }

    array decode(array x, RecurrentKVCache& cache, int max_context) const {
        if (cache.wide.size() != wide_blocks.size())
            cache.wide.resize(wide_blocks.size());
        if (cache.loops.size() != static_cast<size_t>(act.max_loops))
            cache.loops.resize(act.max_loops);
        for (size_t i = 0; i < wide_blocks.size(); ++i)
            x = wide_blocks[i]->decode(x, cache.wide[i]);

        std::vector<array> hs;
        hs.reserve(act.max_loops);
        array memory = zeros_like(x);
        for (int t = 0; t < act.max_loops; ++t) {
            array n_a = add(n_attn(x), multiply(memory, array(0.1f, float16)));
            x = add(x, loop_embed(t, 1));
            array a_out = (t % 2 == 0)
                ? attn_sw.decode(n_a, cache.loops[t])
                : attn_global.decode(n_a, cache.loops[t], max_context);
            array mid = add(x, a_out);
            x = add(mid, moe(n_moe(mid)));
            memory = add(multiply(array(memory_alpha, float16), memory),
                         multiply(array(1.0f - memory_alpha, float16), x));
            hs.push_back(x);
        }
        return act(hs);
    }
};

struct OpenMythos {
    ModelParams mp;
    Linear lm_head;
    RecurrentBlock recurrent;
    RMSNorm final_norm;
    bool elf_enabled = false;
    int elf_latent_dim = 0;
    std::unique_ptr<XavierLinear> elf_self_cond_proj;
    std::unique_ptr<XavierLinearNoBias> elf_bottleneck_in;
    std::unique_ptr<XavierLinear> elf_bottleneck_out;
    std::unique_ptr<Linear> elf_time_in;
    std::unique_ptr<Linear> elf_time_out;
    std::unique_ptr<Linear> elf_cfg_in;
    std::unique_ptr<Linear> elf_cfg_out;
    std::unique_ptr<array> elf_time_tokens;
    std::unique_ptr<array> elf_cfg_tokens;
    std::unique_ptr<array> elf_mode_tokens;
    std::unique_ptr<Linear> elf_output;
    std::unique_ptr<Linear> elf_decoder_proj;
    std::unique_ptr<Linear> elf_decoder_vocab;
    std::optional<array> elf_unembedding;
    float elf_latent_mean = 0.0f;
    float elf_latent_std = 1.0f;

    OpenMythos(int vocab, int dim, int ml, int ws, int gk, int ne, int tk, int nw, int seq_len,
               float capacity_factor = 1.4f, float router_noise_std = 0.0f,
               bool enable_elf = false, int elf_latent_dimension = 0)
        : lm_head(dim, vocab, mp),
          recurrent(dim, ml, ws, gk, ne, tk, nw,
                    seq_len + (enable_elf ? ELF_CONDITION_TOKENS : 0),
                    mp, capacity_factor, router_noise_std),
          final_norm(dim, mp), elf_enabled(enable_elf),
          elf_latent_dim(elf_latent_dimension > 0 ? elf_latent_dimension : dim) {
        if (elf_enabled) {
            elf_self_cond_proj = std::make_unique<XavierLinear>(
                2 * elf_latent_dim, elf_latent_dim, mp);
            elf_bottleneck_in = std::make_unique<XavierLinearNoBias>(
                elf_latent_dim, ELF_BOTTLENECK_DIM, mp);
            elf_bottleneck_out = std::make_unique<XavierLinear>(
                ELF_BOTTLENECK_DIM, dim, mp);
            elf_time_in = std::make_unique<Linear>(ELF_TIME_EMBED_DIM, dim, mp);
            elf_time_out = std::make_unique<Linear>(dim, dim, mp);
            elf_cfg_in = std::make_unique<Linear>(ELF_TIME_EMBED_DIM, dim, mp);
            elf_cfg_out = std::make_unique<Linear>(dim, dim, mp);
            elf_time_tokens = std::make_unique<array>(astype(random::normal(
                {1, ELF_TIME_TOKENS, dim}, 0.0f, 0.02f), float16));
            elf_cfg_tokens = std::make_unique<array>(astype(random::normal(
                {1, ELF_CFG_TOKENS, dim}, 0.0f, 0.02f), float16));
            elf_mode_tokens = std::make_unique<array>(astype(random::normal(
                {1, ELF_MODE_TOKENS, dim}, 0.0f, 0.02f), float16));
            mp.register_param(*elf_time_tokens);
            mp.register_param(*elf_cfg_tokens);
            mp.register_param(*elf_mode_tokens);
            elf_output = std::make_unique<Linear>(dim, elf_latent_dim, mp);
            elf_decoder_proj = std::make_unique<Linear>(dim, elf_latent_dim, mp);
            elf_decoder_vocab = std::make_unique<Linear>(elf_latent_dim, vocab, mp);
            if (elf_latent_dim != dim) {
                // Match official ELF's zero-initialized direct x0 prediction.
                elf_output->W = zeros({dim, elf_latent_dim}, float16);
                elf_output->b = zeros({elf_latent_dim}, float16);
            }
        }
    }

    array token_embeddings(const array& ids) const {
        return take(transpose(lm_head.W, {1, 0}), ids, 0);
    }

    array contextual_token_embeddings(const array& ids, float neighbor_mix) const {
        array base = token_embeddings(ids);
        if (neighbor_mix <= 0.0f) return base;
        int batches = packed_batch_size > 1 ? packed_batch_size : 1;
        int length = ids.shape(0) / batches;
        int dim = base.shape(1);
        array shaped = reshape(base, {batches, length, dim});
        array left = concatenate({
            slice(shaped, {0, 0, 0}, {batches, 1, dim}),
            slice(shaped, {0, 0, 0}, {batches, length - 1, dim})}, 1);
        array right = concatenate({
            slice(shaped, {0, 1, 0}, {batches, length, dim}),
            slice(shaped, {0, length - 1, 0}, {batches, length, dim})}, 1);
        array contextual = add(
            multiply(array(1.0f - 2.0f * neighbor_mix, base.dtype()), shaped),
            multiply(array(neighbor_mix, base.dtype()), add(left, right)));
        return reshape(contextual, {ids.shape(0), dim});
    }

    array operator()(const array& ids) const {
        array x = token_embeddings(ids);
        array h = recurrent(x);
        return lm_head(final_norm(h));
    }

    array denoise_embeddings(
        const array& z, const array& t,
        const std::optional<array>& self_condition = std::nullopt,
        const std::optional<array>& model_mode = std::nullopt,
        std::optional<array>* decoder_logits = nullptr,
        const std::optional<array>& self_cond_cfg_scale = std::nullopt) const {
        if (!elf_enabled)
            throw std::logic_error("ELF denoiser is not enabled for this model");
        int batches = packed_batch_size > 1 ? packed_batch_size : 1;
        int length = z.shape(0) / batches;
        array previous = self_condition ? *self_condition : zeros_like(z);
        array joint = (*elf_self_cond_proj)(concatenate({z, previous}, -1));
        array x = (*elf_bottleneck_out)((*elf_bottleneck_in)(joint));

        auto scalar_embedding = [&](const array& values) {
            const int half = ELF_TIME_EMBED_DIM / 2;
            array indices = astype(arange(half), float32);
            array freqs = exp(multiply(
                array(-std::log(10000.0f) / static_cast<float>(half), float32),
                indices));
            array args = multiply(reshape(astype(values, float32), {batches, 1}),
                                  reshape(freqs, {1, half}));
            return concatenate({cos(args), sin(args)}, -1);
        };

        array t_hidden = (*elf_time_in)(scalar_embedding(t));
        t_hidden = multiply(t_hidden, sigmoid(t_hidden));
        t_hidden = (*elf_time_out)(t_hidden);
        array mode_values = model_mode ? *model_mode : zeros_like(t);
        array cfg_values = self_cond_cfg_scale
            ? *self_cond_cfg_scale : zeros_like(t);
        array cfg_hidden = (*elf_cfg_in)(scalar_embedding(cfg_values));
        cfg_hidden = multiply(cfg_hidden, sigmoid(cfg_hidden));
        cfg_hidden = (*elf_cfg_out)(cfg_hidden);

        array time_prefix = add(
            broadcast_to(*elf_time_tokens, {batches, ELF_TIME_TOKENS, x.shape(1)}),
            broadcast_to(expand_dims(astype(t_hidden, x.dtype()), 1),
                         {batches, ELF_TIME_TOKENS, x.shape(1)}));
        array cfg_prefix = add(
            broadcast_to(*elf_cfg_tokens, {batches, ELF_CFG_TOKENS, x.shape(1)}),
            broadcast_to(expand_dims(astype(cfg_hidden, x.dtype()), 1),
                         {batches, ELF_CFG_TOKENS, x.shape(1)}));
        array mode_gate = reshape(astype(mode_values, x.dtype()), {batches, 1, 1});
        array mode_prefix = multiply(
            broadcast_to(*elf_mode_tokens, {batches, ELF_MODE_TOKENS, x.shape(1)}),
            broadcast_to(mode_gate, {batches, ELF_MODE_TOKENS, x.shape(1)}));
        array x_batched = reshape(x, {batches, length, x.shape(1)});
        array with_context = concatenate(
            {mode_prefix, time_prefix, cfg_prefix, x_batched}, 1);
        array h_all = recurrent(reshape(
            with_context,
            {batches * (length + ELF_CONDITION_TOKENS), x.shape(1)}));
        array h = reshape(slice(
            reshape(h_all,
                    {batches, length + ELF_CONDITION_TOKENS, x.shape(1)}),
            {0, ELF_CONDITION_TOKENS, 0},
            {batches, length + ELF_CONDITION_TOKENS, x.shape(1)}),
            {z.shape(0), x.shape(1)});
        if (decoder_logits) {
            array projected = (*elf_decoder_proj)(h);
            // GELU, matching the factored decoder head in official ELF.
            array activated = multiply(multiply(array(0.5f, projected.dtype()), projected),
                add(array(1.0f, projected.dtype()), erf(divide(projected,
                    array(std::sqrt(2.0f), projected.dtype())))));
            *decoder_logits = (*elf_decoder_vocab)(activated);
        }
        // Official ELF directly predicts x0; the zero-initialized final layer
        // therefore starts at x=0 rather than as a residual identity map.
        return (*elf_output)(final_norm(h));
    }

    array decode_embeddings(const array& embeddings) const {
        if (elf_unembedding) {
            array raw_embeddings = add(
                multiply(embeddings, array(elf_latent_std, embeddings.dtype())),
                array(elf_latent_mean, embeddings.dtype()));
            array scaled = multiply(
                raw_embeddings,
                array(1.0f / std::sqrt(static_cast<float>(elf_latent_dim)),
                      embeddings.dtype()));
            return matmul(scaled, transpose(*elf_unembedding, {1, 0}));
        }
        array projected = (*elf_bottleneck_out)((*elf_bottleneck_in)(embeddings));
        return lm_head(projected);
    }

    void set_elf_unembedding(const array& weights) {
        if (!elf_enabled || weights.ndim() != 2 ||
            weights.shape(1) != elf_latent_dim)
            throw std::invalid_argument("ELF unembedding shape mismatch");
        elf_unembedding = stop_gradient(weights);
    }

    void set_elf_latent_normalization(float mean, float stddev) {
        if (stddev <= 0.0f)
            throw std::invalid_argument("ELF latent std must be positive");
        elf_latent_mean = mean;
        elf_latent_std = stddev;
    }

    array decode(const array& id, RecurrentKVCache& cache, int max_context) const {
        array embed = transpose(lm_head.W, {1, 0});
        array x = take(embed, id, 0);
        bool previous_decode_state = incremental_decode_active;
        incremental_decode_active = true;
        array h = recurrent.decode(x, cache, max_context);
        incremental_decode_active = previous_decode_state;
        return lm_head(final_norm(h));
    }

#ifdef USE_CAPACITY_MOE
    // 切换所有 CapacityMoE 实例（顶层 recurrent.moe + 每个 wide_blocks[i]->moe）
    // 的训练/推理模式。训练时 is_training=true 启用路由噪声，推理/生成阶段
    // 应该调用 set_training(false) 关闭噪声，确保路由判断是确定性的、用模型
    // 学到的最优策略，而不是带着随机探索性的训练状态。
    void set_training(bool training) const {
        recurrent.moe.is_training = training;
        for (auto& blk : recurrent.wide_blocks) blk->moe.is_training = training;
    }
#endif

    // 调试辅助：从真实token ids正确构造embedding后，统计顶层RecurrentBlock.moe
    // 的专家负载分布。之前的监控代码错误地直接把1D的token id数组传给了
    // debug_fill_count（它内部的router期望[L,dim]浮点embedding，不是1D整数id），
    // 这里改为先做正确的embedding lookup，避免[slice] Invalid number of indices
    // 这类形状不匹配的崩溃。
    //
    // 加 #ifdef 保护：BatchedMoE 没有 debug_fill_count 方法，如果切回原版
    // （注释掉 USE_CAPACITY_MOE）但忘记同时去掉这个方法的调用点，会编译失败。
#ifdef USE_CAPACITY_MOE
    array debug_moe_fill_count(const array& ids) const {
        array embed = transpose(lm_head.W, {1, 0});
        array x = take(embed, ids, 0);
        return recurrent.moe.debug_fill_count(x);
    }

    // 汇总整个模型里所有 CapacityMoE 实例（顶层 recurrent.moe + 每个
    // wide_blocks[i]->moe）在本次forward中产生的负载均衡辅助损失。
    // 这些 aux_loss 是在 RecurrentBlock/TransformerBlock::operator() 内部
    // 通过 mutable 成员"side channel"方式记录的，使用的是forward时
    // 真实的中间输入（不是另外重新跑一遍的近似值）。
    array total_aux_loss() const {
        array total = recurrent.aux_loss;
        for (auto& blk : recurrent.wide_blocks) total = add(total, blk->aux_loss);
        return total;
    }
#endif
};

// ============================================================================
// 6. 辅助函数
// ============================================================================

// ----------------------------------------------------------------------------
// 终端可视化工具：sparkline 走势图 + 横向柱状图
// 全部用 Unicode block 字符手写实现，零外部依赖，纯标准库。
// ----------------------------------------------------------------------------

// 把一组数值渲染成一行 sparkline（用8级block字符 ▁▂▃▄▅▆▇█ 表示相对高低）
// 用于在终端里快速看出 loss / 梯度范数等指标的走势趋势，而不只是孤立数字。
// 把任意长度的数据分桶平均，压缩/降采样到固定的 target_width 个点。
// 这样不管底层历史缓冲攒了多少数据，sparkline 显示宽度永远恒定，
// 不会出现"越打越长"的问题（之前的版本直接把整个历史塞进sparkline，
// 长度会一直增长到 RollingHistory 的capacity才封顶，过程中很乱）。
std::vector<float> downsample(const std::vector<float>& values, int target_width) {
    if ((int)values.size() <= target_width) return values;
    std::vector<float> out(target_width, 0.0f);
    float bucket_size = static_cast<float>(values.size()) / target_width;
    for (int i = 0; i < target_width; ++i) {
        int start = static_cast<int>(i * bucket_size);
        int end = static_cast<int>((i + 1) * bucket_size);
        end = std::max(end, start + 1);
        end = std::min(end, (int)values.size());
        float sum = 0.0f;
        for (int j = start; j < end; ++j) sum += values[j];
        out[i] = sum / (end - start);
    }
    return out;
}

std::string render_sparkline(const std::vector<float>& values, int target_width = 30) {
    static const char* blocks[8] = {
        "\u2581", "\u2582", "\u2583", "\u2584",
        "\u2585", "\u2586", "\u2587", "\u2588"
    };
    if (values.empty()) return "";
    std::vector<float> ds = downsample(values, target_width);

    float vmin = ds[0], vmax = ds[0];
    for (float v : ds) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    float range = vmax - vmin;
    if (range < 1e-8f) range = 1e-8f;  // 避免全部相同时除零

    std::string out;
    for (float v : ds) {
        int level = static_cast<int>(((v - vmin) / range) * 7.0f + 0.5f);
        level = std::max(0, std::min(7, level));
        out += blocks[level];
    }
    return out;
}

// 把一组数值（通常是各专家的负载数）渲染成横向柱状图，每个值占一行，
// 柱子长度按 max_bar_width 个字符等比缩放。用于直观看出专家负载是否均衡——
// 如果某一行明显比其他行长很多，说明路由开始偏向某个专家（专家坍缩信号）。
std::string render_horizontal_bars(const std::vector<float>& values, int max_bar_width = 30) {
    if (values.empty()) return "";
    float vmax = values[0];
    for (float v : values) vmax = std::max(vmax, v);
    if (vmax < 1e-8f) vmax = 1.0f;

    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        int bar_len = static_cast<int>((values[i] / vmax) * max_bar_width + 0.5f);
        bar_len = std::max(0, std::min(max_bar_width, bar_len));
        oss << "    E" << i << " ";
        if (i < 10) oss << " ";  // 个位数专家编号对齐
        oss << "[";
        for (int j = 0; j < bar_len; ++j) oss << "\u2588";
        for (int j = 0; j < max_bar_width - bar_len; ++j) oss << "\u00b7";
        oss << "] "
            << std::fixed << std::setprecision(0) << values[i] << "\n";
    }
    return oss.str();
}

// 滚动历史缓冲：固定容量的队列，超出容量自动丢弃最旧的值。
// 用于维护"最近N步"的loss/梯度范数历史，喂给sparkline渲染。
struct RollingHistory {
    std::deque<float> data;
    size_t capacity;
    RollingHistory(size_t cap) : capacity(cap) {}
    void push(float v) {
        data.push_back(v);
        if (data.size() > capacity) data.pop_front();
    }
    std::vector<float> to_vector() const {
        return std::vector<float>(data.begin(), data.end());
    }
};

array cross_entropy_loss(const array& logits, const array& targets, int ignore_index = -100, float entropy_beta = 0.01f) {
    array logits_f32 = astype(logits, float32);

    array max_logits = max(logits_f32, -1, true);               
    array shifted = subtract(logits_f32, max_logits);
    array exp_shifted = exp(shifted);
    array sum_exp = sum(exp_shifted, -1, true);
    array log_probs = subtract(shifted, log(sum_exp));

    array gathered = take_along_axis(log_probs, expand_dims(targets, -1), -1);
    gathered = reshape(gathered, {-1});                     
    array mask = not_equal(targets, array(ignore_index, targets.dtype()));
    array mask_f = astype(mask, float32);   
    array base_loss = multiply(negative(gathered), mask_f);

    array probs = divide(exp_shifted, sum_exp);
    array entropy = multiply(negative(probs), log_probs); 
    array token_entropy = sum(entropy, -1); 
    array confidence_penalty = multiply(base_loss, exp(negative(token_entropy))); 

    array total_token_loss = add(base_loss, multiply(array(entropy_beta, float32), confidence_penalty));

    // 优化：denom 不再 .item<float>() 同步回 host——那会在 value_and_grad
    // 内部每次前向都强制一次 device→host 同步（每个 micro-step 一次），
    // 严重打乱 GPU 流水线。total_token_loss 已被 mask_f 掩码过，所以
    // sum(total_token_loss) / sum(mask_f) 与原来的 (sum/denom) 数值完全
    // 一致，且整个除法留在计算图里，只在训练循环结束时才同步一次标量。
    if (packed_batch_size > 1) {
        int sequence_length = targets.shape(0) / packed_batch_size;
        array per_sequence_loss = sum(
            reshape(total_token_loss, {packed_batch_size, sequence_length}), 1);
        array per_sequence_tokens = add(sum(
            reshape(mask_f, {packed_batch_size, sequence_length}), 1),
            array(1e-8f, float32));
        return mean(divide(per_sequence_loss, per_sequence_tokens));
    }
    array denom = add(sum(mask_f), array(1e-8f, float32));
    return divide(sum(total_token_loss), denom);
}

int sample_from_probs(const array& logits, float temp = 1.0f) {
    array scaled = divide(astype(logits, float32), array(temp, float32));
    array probs = softmax(scaled, -1);
    eval({probs}); 
    
    const float* p_data = probs.data<float>();
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    float r = unif(rng), cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) { 
        cum += p_data[i]; 
        if (r <= cum) return i; 
    }
    return probs.size() - 1;
}

float get_lr(int step, int total_steps, float peak_lr, int warmup_steps) {
    if (step < warmup_steps) return peak_lr * (step + 1) / warmup_steps;
    float progress = float(step - warmup_steps) / (total_steps - warmup_steps);
    if (progress > 1.0f) progress = 1.0f;
    return peak_lr * 0.5f * (1.0f + std::cos(M_PI * progress));
}

// ============================================================================
// 7. 训练主逻辑
// ============================================================================
static float parse_finite_float(const std::string &text) {
    size_t end = 0;
    const float value = std::stof(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::invalid_argument("Invalid finite numeric option: " + text);
    return value;
}

int main(int argc, char* argv[]) {
  try {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help") {
        std::cout << "turtle-mlx [gen PROMPT | eval] [options]\n"
                     "--device auto|cpu|gpu  --data-dir PATH  --steps N  --model-mode ar|elf\n"
                     "--dim N (multiple of 16) --seq-len N --max-loop N --wide-blocks N\n"
                     "--experts N --moe-topk N --window-size N --global-topk N\n"
                     "--model-out PATH --tokenizer-out PATH (training)\n"
                     "--model PATH --tokenizer PATH --max-tokens N (generation)\n"
                     "--eval-data PATH --eval-samples N --eval-untrained 0|1 (AR evaluation)\n"
                     "--optimizer-memory auto|resident|swap --swap-dir PATH --galore-off 0|1\n"
                     "See mlx/README.md for ELF/T5, GaLore and optimization flags.\n";
        return 0;
    }

    // ---------- 超参数 ----------
    int dim = 192;              
    int max_loop = 4;          
    int window_size = 128;      
    int global_topk = 32;      
    int seq_len = 192;          
    int accum = 4;            
    int moe_experts = 8;       
    int moe_topk = 2;          // 核心提速：Top-1 路由
    int num_wide_blocks = 3;   
    int total_steps = 5000;
    
    // 🌟 核心修复：调大峰值学习率并大幅缩短 Warmup，强行撕开拟合突破口
    float peak_lr = 3e-4f;

    auto get_arg = [&](const std::string& key, const std::string& def) -> std::string {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == key) return argv[i + 1];
        return def;
    };
    auto get_int_arg = [&](const std::string& key, int def) -> int {
        std::string v = get_arg(key, "");
        return v.empty() ? def : std::stoi(v);
    };
    const std::string device = get_arg("--device", "auto");
    if (device != "auto" && device != "cpu" && device != "gpu") throw std::invalid_argument("--device must be auto, cpu or gpu");
    const bool have_gpu = is_available(Device(Device::gpu));
    if (device == "gpu" && !have_gpu) throw std::runtime_error("MLX GPU backend is unavailable; use --device cpu");
    set_default_device(Device(device == "cpu" || (device == "auto" && !have_gpu) ? Device::cpu : Device::gpu));
    std::cout << "OpenMythos v6.4 / turtle-mlx on " << (default_device() == Device::cpu ? "CPU" : "GPU") << std::endl;
    dim = get_int_arg("--dim", dim);
    seq_len = get_int_arg("--seq-len", seq_len);
    max_loop = get_int_arg("--max-loop", max_loop);
    window_size = get_int_arg("--window-size", window_size);
    global_topk = get_int_arg("--global-topk", global_topk);
    moe_experts = get_int_arg("--experts", moe_experts);
    moe_topk = get_int_arg("--moe-topk", moe_topk);
    num_wide_blocks = get_int_arg("--wide-blocks", num_wide_blocks);
    total_steps = get_int_arg("--steps", total_steps);
    if (dim <= 0 || dim % 16 != 0 || seq_len <= 0 || max_loop <= 0 || window_size <= 0 || global_topk <= 0 ||
        moe_experts <= 0 || moe_topk <= 0 || moe_topk > moe_experts || num_wide_blocks < 0 || total_steps <= 0)
        throw std::invalid_argument("Invalid model dimensions, expert count or training steps");
    optimizer_memory_mode = get_arg("--optimizer-memory", "auto");
    if (optimizer_memory_mode != "auto" && optimizer_memory_mode != "resident" && optimizer_memory_mode != "swap")
        throw std::invalid_argument("Invalid --optimizer-memory");
    peak_lr = parse_finite_float(get_arg("--peak-lr", std::to_string(peak_lr)));
    if (!std::isfinite(peak_lr) || peak_lr <= 0.0f)
        throw std::invalid_argument("--peak-lr must be positive");
    accum = get_int_arg("--batch-size", accum);
    if (accum <= 0) throw std::invalid_argument("--batch-size must be positive");
    random::seed(get_int_arg("--seed", 42));
    resident_optimizer = get_int_arg("--resident-optimizer", 1) != 0;
    fused_moe_aux = get_int_arg("--fused-moe-aux", 1) != 0;
    fast_rmsnorm = get_int_arg("--fast-rmsnorm", 1) != 0;
    batch_galore_probes = get_int_arg("--batch-galore-probes", 1) != 0;
    partition_moe_topk = get_int_arg("--partition-moe-topk", 1) != 0;
    gather_sparse_attention = get_int_arg("--gather-sparse-attention", 0) != 0;
    fused_qkv_projection = get_int_arg("--fused-qkv", 1) != 0;
    fast_decode_sdpa = get_int_arg("--fast-decode-sdpa", 0) != 0;
    vectorized_sparse_decode = get_int_arg("--vectorized-sparse-decode", 1) != 0;
    rng.seed(get_int_arg("--seed", 42));
    const int cache_mb = get_int_arg("--cache-limit-mb", 512);
    const int clear_interval = get_int_arg("--clear-cache-interval", 0);
    bool use_compiled_train_graph = get_int_arg("--compile-train-graph", 1) != 0;
    const bool use_packed_batch = get_int_arg("--packed-batch", 1) != 0;
    const std::string model_mode = get_arg("--model-mode", "ar");
    const bool elf_mode = model_mode == "elf";
    if (!elf_mode && model_mode != "ar")
        throw std::invalid_argument("--model-mode must be ar or elf");
    bidirectional_training_attention = elf_mode;
    rope_empty_prefix_tokens = elf_mode ? ELF_CONDITION_TOKENS : 0;
    if (elf_mode && gather_sparse_attention) {
        std::cout << "⚠️ ELF condition tokens require dense-score sparse attention; "
                     "disabling --gather-sparse-attention.\n";
        gather_sparse_attention = false;
    }
    const std::string elf_t5_latents_path = get_arg("--elf-t5-latents", "");
    const std::string elf_t5_unembedding_path = get_arg("--elf-t5-unembedding", "");
    std::unique_ptr<T5LatentDataset> elf_t5_dataset;
    std::unique_ptr<T5Unembedding> elf_t5_unembedding;
    if (elf_mode && !elf_t5_latents_path.empty())
        elf_t5_dataset = std::make_unique<T5LatentDataset>(elf_t5_latents_path);
    if (elf_mode && !elf_t5_unembedding_path.empty())
        elf_t5_unembedding = std::make_unique<T5Unembedding>(elf_t5_unembedding_path);
    const int elf_latent_dim = elf_t5_dataset
        ? static_cast<int>(elf_t5_dataset->latent_dim)
        : get_int_arg("--elf-latent-dim", dim);
    const float elf_latent_mean = parse_finite_float(get_arg("--elf-latent-mean", "0.0"));
    const float elf_latent_std = parse_finite_float(get_arg(
        "--elf-latent-std", elf_t5_dataset ? "0.2" : "1.0"));
    const float elf_denoiser_noise_scale = parse_finite_float(get_arg(
        "--elf-denoiser-noise-scale", elf_t5_dataset ? "2.0" : "1.0"));
    if (elf_latent_dim <= 0)
        throw std::invalid_argument("--elf-latent-dim must be positive");
    if (elf_latent_std <= 0.0f || elf_denoiser_noise_scale <= 0.0f)
        throw std::invalid_argument("Invalid ELF latent/noise scale");
    if (elf_t5_unembedding &&
        static_cast<int>(elf_t5_unembedding->latent_dim) != elf_latent_dim)
        throw std::invalid_argument("T5 latent/unembedding dimension mismatch");
    const uint32_t elf_checkpoint_magic = ELF_PREFIX_CHECKPOINT_MAGIC;
    if (cache_mb < 0 || clear_interval < 0) throw std::invalid_argument("Invalid cache configuration");
    set_cache_limit(static_cast<size_t>(cache_mb) * 1024 * 1024);

    // ---------- GaLore 命令行配置 ----------
    // --galore-off 1          : 完全关闭，回到原版全量优化器（A/B对照实验用）
    // --galore-rank N         : 子空间秩，默认128。越大越接近全量行为但内存省得越少；
    //                           经验起点：过小(如16)可能减慢收敛，128对本模型的
    //                           MoE专家矩阵（reshape后1536x768）是合理量级
    // --galore-refresh N      : 兜底最大刷新间隔，默认300。不管动态信号是否触发，
    //                           超过这个步数也强制重建一次，防止信号本身长期失效。
    //                           若配合 --galore-fixed-refresh 1，则此值就是固定间隔本身。
    // --galore-min-interval N : 最短刷新间隔，默认10。刚重建后这么多步内忽略触发信号，
    //                           防止梯度噪声导致刷新过于频繁（每步都刷=白投影）。
    // --galore-cos-thresh F   : 余弦相似度阈值，默认0.5。当前梯度方向与构建子空间
    //                           时的参考梯度方向的余弦相似度低于此值，判定子空间已
    //                           过时，触发重建。诊断实验：固定refresh=200时训练最
    //                           初200步几乎全程用同一子空间硬扛，loss从baseline的
    //                           3.75劣化到5.26；改余弦触发后前50步loss从0.157改善
    //                           到0.092，最终质量也从0.0175提升到0.0130附近。
    // --galore-fixed-refresh 1: 退化为纯固定间隔模式(=--galore-refresh的值)，
    //                           不使用余弦信号。用于排查问题时快速切回旧行为。
    // --galore-min-size N     : 参数量低于N的张量不投影，默认65536。bias/norm等
    //                           小参数投影没有内存收益，反而多付矩阵乘法开销
    // 注意：rank 和 min-size 影响 checkpoint 内的状态布局，已持久化到 ckpt 中，
    // 断点恢复时以 ckpt 里存的为准（避免CLI不一致导致读错数据）。触发相关的
    // min-interval/cos-thresh/fixed-refresh 不影响 packed 布局，不持久化，
    // 可以在断点恢复时自由调整实验。
    const bool  galore_off        = get_int_arg("--galore-off", 0) != 0;
    const int   galore_rank_cli   = get_int_arg("--galore-rank", 128);
    const int   galore_refresh_cli    = get_int_arg("--galore-refresh", 300);
    const int   galore_mininterval_cli = get_int_arg("--galore-min-interval", 10);
    const float galore_costhresh_cli  = parse_finite_float(get_arg("--galore-cos-thresh", "0.5"));
    const bool  galore_fixedmode_cli  = get_int_arg("--galore-fixed-refresh", 0) != 0;
    const int   galore_minsz_cli   = get_int_arg("--galore-min-size", 65536);
    if (galore_rank_cli <= 0 || galore_refresh_cli <= 0 || galore_mininterval_cli <= 0 ||
        galore_minsz_cli <= 0 || !std::isfinite(galore_costhresh_cli) || galore_costhresh_cli < -1 || galore_costhresh_cli > 1)
        throw std::invalid_argument("Invalid GaLore configuration");
    auto metadata_for = [&](const BPETrainer &tokenizer, int vocab) {
        return nlohmann::json{{"format_version", 1}, {"backend", "mlx-openmythos"},
            {"byte_order", checkpoint::byte_order()}, {"dim", dim}, {"seq_len", seq_len}, {"max_loop", max_loop},
            {"window_size", window_size}, {"global_topk", global_topk}, {"moe_experts", moe_experts},
            {"moe_topk", moe_topk}, {"wide_blocks", num_wide_blocks}, {"model_mode", model_mode},
            {"elf_latent_dim", elf_latent_dim}, {"vocab_size", vocab},
            {"tokenizer_fingerprint", elf_t5_dataset ? "t5-external" : tokenizer.fingerprint()},
#ifdef USE_CAPACITY_MOE
            {"capacity_moe", true}
#else
            {"capacity_moe", false}
#endif
        };
    };
    auto apply_galore_cfg = [&](ModelParams& mp) {
        mp.swap_dir = get_arg("--swap-dir", "ssd_swap") + "/";
        mp.galore_enabled      = !galore_off;
        mp.galore_rank         = galore_rank_cli;
        mp.galore_refresh      = galore_refresh_cli;
        mp.galore_min_interval = galore_mininterval_cli;
        mp.galore_cos_thresh   = galore_costhresh_cli;
        mp.galore_fixed_refresh = galore_fixedmode_cli;
        mp.galore_min_size    = (size_t)galore_minsz_cli;
    };

    // Generation and read-only AR evaluation share validated model loading.
    const bool evaluate = argc >= 2 && std::string(argv[1]) == "eval";
    const bool evaluate_untrained = evaluate && get_int_arg("--eval-untrained", 0) != 0;
    if (evaluate && elf_mode)
        throw std::invalid_argument("eval currently supports AR models only");
    if (evaluate || (argc >= 3 && std::string(argv[1]) == "gen")) {
        std::string model_path = get_arg(
            "--model", elf_mode ? "openmythos_elf7.ckpt" : "openmythos_mlx.ckpt");
        std::string tok_path   = get_arg("--tokenizer", "tokenizer.bpe");
        int max_tokens         = get_int_arg("--max-tokens", 200);
        float temp             = parse_finite_float(get_arg("--temp", "0.7"));
        int context_length     = get_int_arg("--context-length", seq_len);
        int gen_clear_interval = get_int_arg("--gen-clear-cache-interval", 0);
        bool use_kv_cache      = get_int_arg("--kv-cache", 1) != 0;
        if (context_length <= 0 || gen_clear_interval < 0)
            throw std::invalid_argument("Invalid generation cache configuration");

        if (!std::filesystem::is_regular_file(model_path)) throw std::runtime_error("Model checkpoint not found: " + model_path);
        if (max_tokens < 0 || !std::isfinite(temp) || temp <= 0) throw std::invalid_argument("Invalid generation length or temperature");
        BPETrainer tokenizer;
        const bool t5_latent_generation = elf_mode && elf_t5_dataset &&
            elf_t5_unembedding && elf_latent_dim != dim;
        if (!t5_latent_generation && !tokenizer.load(tok_path)) {
            std::cerr << "ERROR: tokenizer not found" << std::endl;
            return 1;
        }
        // 注：CapacityMoE 的 capacity 是基于训练时 seq_len 算出的固定值，
        // 不会随生成阶段实际 token 数 L 自动调整。这意味着：
        //   - L 较小时（生成刚开始）：capacity 有冗余，纯粹是计算浪费，不影响正确性
        //   - L 较大时：如果路由恰好让某个专家被过多token选中（分配数超过
        //     capacity），超出部分会被丢弃（不会崩溃，但那些token在MoE这一层
        //     的贡献会缺失，可能轻微影响生成质量）。这是capacity机制本身的
        //     已知限制，不是bug。如果你的生成长度会持续大幅超过训练seq_len，
        //     建议用 debug_fill_count 在生成时抽样监控是否频繁溢出。
        int generation_vocab = elf_t5_unembedding
            ? static_cast<int>(elf_t5_unembedding->vocab_size)
            : static_cast<int>(tokenizer.vocab_size());
        OpenMythos model(generation_vocab, dim, max_loop, window_size, global_topk,
                         moe_experts, moe_topk, num_wide_blocks, seq_len,
                         1.4f, 0.0f, elf_mode, elf_latent_dim);
        if (elf_t5_unembedding)
            model.set_elf_unembedding(elf_t5_unembedding->weights);
        if (elf_t5_dataset)
            model.set_elf_latent_normalization(elf_latent_mean, elf_latent_std);
#ifdef USE_CAPACITY_MOE
        // 推理/生成阶段关闭路由噪声：噪声是训练时用来打破专家坍缩的探索性
        // 机制，推理时不应该再随机扰动路由判断——这里用的是模型已经学到的
        // 路由策略，应该是确定性的，加噪声只会增加结果方差，没有任何好处。
        model.set_training(false);
#endif
        // GaLore配置需在load之前注入（load内部会调init_swap分配packed缓冲；
        // 布局相关的rank/min-size随后会被ckpt里持久化的值覆盖，保证一致）
        apply_galore_cfg(model.mp);
        {
            std::ifstream is(model_path, std::ios::binary);
            if (is) {
                mlx_checkpoint::read_metadata(is, metadata_for(tokenizer, generation_vocab));
                int dummy_step = 0;
                is.read(reinterpret_cast<char*>(&dummy_step), sizeof(int));
                if (elf_mode) {
                    uint32_t magic = 0;
                    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    if (magic != elf_checkpoint_magic) {
                        std::cerr << "ERROR: checkpoint is not an ELF checkpoint\n";
                        return 1;
                    }
                }
                if (!evaluate_untrained && !model.mp.load(is)) {
                    std::cerr << "ERROR: checkpoint parameter layout mismatch\n";
                    return 1;
                }
            }
        }

        if (evaluate) {
            const std::string eval_path = get_arg("--eval-data", "");
            const int requested_samples = get_int_arg("--eval-samples", 64);
            if (eval_path.empty() || requested_samples <= 0)
                throw std::invalid_argument("eval requires --eval-data and positive --eval-samples");
            TrainingCorpus validation = read_training_corpus(eval_path);
            std::vector<size_t> indices(validation.texts.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            packed_batch_size = 1;
            double negative_log_likelihood = 0.0;
            size_t token_count = 0;
            int samples = 0;
            for (size_t index : indices) {
                auto tokens = tokenizer.encode(validation.texts[index]);
                if (tokens.empty()) continue;
                tokens.insert(tokens.begin(), FILE_START_TOKEN_ID);
                tokens.push_back(FILE_END_TOKEN_ID);
                const size_t length = std::min(tokens.size() - 1, static_cast<size_t>(seq_len));
                const size_t max_start = tokens.size() - length - 1;
                const size_t start = std::uniform_int_distribution<size_t>(0, max_start)(rng);
                std::vector<int> input(tokens.begin() + start, tokens.begin() + start + length);
                std::vector<int> target(tokens.begin() + start + 1, tokens.begin() + start + length + 1);
                const size_t valid_tokens = std::count_if(target.begin(), target.end(),
                    [](int token) { return token != PAD_TOKEN_ID; });
                if (!valid_tokens) continue;
                array logits = model(array(input.data(), {static_cast<int>(length)}, int32));
                array targets = array(target.data(), {static_cast<int>(length)}, int32);
                // Pure CE: omit the training confidence penalty and MoE auxiliary loss.
                const float ce = cross_entropy_loss(logits, targets, PAD_TOKEN_ID, 0.0f).item<float>();
                if (!std::isfinite(ce)) throw std::runtime_error("Nonfinite evaluation loss");
                negative_log_likelihood += static_cast<double>(ce) * valid_tokens;
                token_count += valid_tokens;
                if (++samples >= requested_samples) break;
            }
            if (!token_count) throw std::invalid_argument("No usable evaluation tokens");
            const double ce = negative_log_likelihood / token_count;
            const double perplexity = std::exp(ce);
            if (!std::isfinite(perplexity)) throw std::runtime_error("Evaluation perplexity overflow");
            nlohmann::json metrics{{"cross_entropy", ce}, {"perplexity", perplexity},
                {"tokens", token_count}, {"samples", samples}, {"seed", get_int_arg("--seed", 42)},
                {"weights", evaluate_untrained ? "random_initialization" : "checkpoint"},
                {"dataset_identity", validation.identity}, {"context_length", seq_len}};
            std::cout << "EVAL " << metrics.dump() << std::endl;
            return 0;
        }

        std::string prompt = argv[2];
        std::vector<TokenId> ids;
        if (!t5_latent_generation) {
            if (elf_mode) ids = tokenizer.encode(prompt, true);
            else {
                // AR training uses file boundaries. A generation prompt is an
                // unfinished prefix, so do not append EOS before its continuation.
                ids = tokenizer.encode(prompt);
                ids.insert(ids.begin(), FILE_START_TOKEN_ID);
            }
        }
        std::cout << "READY\n" << std::flush;

        if (elf_mode) {
            packed_batch_size = 1;
            bidirectional_training_attention = true;
            int sample_steps = get_int_arg("--elf-sample-steps", 32);
            float sde_gamma = parse_finite_float(get_arg("--elf-sde-gamma", "1.5"));
            const std::string sample_time_schedule = get_arg(
                "--elf-sample-time-schedule", "logit_normal");
            const float sample_p_mean = parse_finite_float(get_arg(
                "--elf-sample-p-mean", "-1.5"));
            const float sample_p_std = parse_finite_float(get_arg(
                "--elf-sample-p-std", "0.8"));
            const float self_cond_cfg_scale = parse_finite_float(get_arg(
                "--elf-self-cond-cfg-scale", "3.0"));
            float context_mix = parse_finite_float(get_arg("--elf-context-mix", "0.15"));
            if (sample_steps <= 0 || max_tokens <= 0)
                throw std::invalid_argument("Invalid ELF sampling configuration");
            if (sample_p_std <= 0.0f || self_cond_cfg_scale < 0.0f ||
                (sample_time_schedule != "uniform" &&
                 sample_time_schedule != "logit_normal"))
                throw std::invalid_argument("Invalid ELF sampling time schedule");
            if (context_mix < 0.0f || context_mix >= 0.5f)
                throw std::invalid_argument("--elf-context-mix must be in [0, 0.5)");
            int prefix_length = 0;
            int output_tokens = 0;
            array clean_prefix = zeros({0, elf_latent_dim}, float16);
            array clean_suffix = zeros({0, elf_latent_dim}, float16);
            std::vector<int32_t> t5_ids;
            std::vector<float16_t> t5_latents;
            if (t5_latent_generation) {
                int record = get_int_arg("--elf-eval-record", 0);
                prefix_length = get_int_arg("--elf-prefix-length", seq_len / 2);
                if (record < 0 || record >= static_cast<int>(elf_t5_dataset->records))
                    throw std::invalid_argument("--elf-eval-record is out of range");
                if (prefix_length <= 0 || prefix_length >= seq_len)
                    throw std::invalid_argument("--elf-prefix-length must be in (0, seq_len)");
                output_tokens = std::min(max_tokens, seq_len - prefix_length);
                t5_ids.resize(seq_len);
                t5_latents.resize(static_cast<size_t>(seq_len) * elf_latent_dim);
                elf_t5_dataset->read_record(
                    static_cast<uint32_t>(record), t5_ids.data(), t5_latents.data());
                array latent_record = array(
                    t5_latents.data(), {seq_len, elf_latent_dim}, float16);
                latent_record = divide(
                    subtract(astype(latent_record, float32),
                             array(elf_latent_mean, float32)),
                    array(elf_latent_std, float32));
                clean_prefix = stop_gradient(slice(
                    latent_record, {0, 0}, {prefix_length, elf_latent_dim}));
                clean_suffix = stop_gradient(slice(
                    latent_record, {prefix_length, 0},
                    {prefix_length + output_tokens, elf_latent_dim}));
                std::cout << "T5_EVAL record=" << record
                          << " prefix=" << prefix_length
                          << " suffix=" << output_tokens
                          << " steps=" << sample_steps << "\n";
            } else {
                output_tokens = std::min(
                    max_tokens, std::max(1, seq_len - 1));
                if (ids.size() + static_cast<size_t>(output_tokens) >
                    static_cast<size_t>(seq_len)) {
                    size_t keep = static_cast<size_t>(seq_len - output_tokens);
                    if (ids.size() > keep)
                        ids = std::vector<TokenId>(ids.end() - keep, ids.end());
                }
                prefix_length = static_cast<int>(ids.size());
                array prefix_ids = array(ids.data(), {prefix_length}, int32);
                clean_prefix = stop_gradient(
                    model.contextual_token_embeddings(prefix_ids, context_mix));
            }
            array suffix_noise = multiply(
                random::normal({output_tokens, elf_latent_dim}),
                array(elf_denoiser_noise_scale, float32));
            float direct_t = parse_finite_float(get_arg("--elf-eval-direct-t", "-1"));
            if (direct_t >= 0.0f) {
                if (!t5_latent_generation || direct_t > 1.0f)
                    throw std::invalid_argument(
                        "--elf-eval-direct-t requires T5 evaluation and t in [0, 1]");
                suffix_noise = add(
                    multiply(array(1.0f - direct_t, float32), suffix_noise),
                    multiply(array(direct_t, float16), clean_suffix));
            }
            array z = concatenate({clean_prefix, astype(suffix_noise, float16)}, 0);
            array x_pred = concatenate({
                clean_prefix,
                zeros({output_tokens, elf_latent_dim}, float16)}, 0);
            array cfg_scale_array = array(&self_cond_cfg_scale, {1}, float32);
            if (direct_t >= 0.0f) {
                array t_array = array(&direct_t, {1}, float32);
                x_pred = model.denoise_embeddings(
                    z, t_array, x_pred, std::nullopt, nullptr,
                    cfg_scale_array);
                eval({x_pred});
            }
            std::vector<float> sample_times(sample_steps + 1);
            sample_times.front() = 0.0f;
            sample_times.back() = 1.0f;
            if (sample_time_schedule == "uniform") {
                for (int i = 1; i < sample_steps; ++i)
                    sample_times[i] = static_cast<float>(i) / sample_steps;
            } else {
                std::normal_distribution<float> sample_time_normal(0.0f, 1.0f);
                for (int i = 1; i < sample_steps; ++i) {
                    float value = sample_p_mean +
                        sample_p_std * sample_time_normal(rng);
                    sample_times[i] = 1.0f / (1.0f + std::exp(-value));
                }
                std::sort(sample_times.begin() + 1, sample_times.end() - 1);
            }
            for (int step = 0; direct_t < 0.0f && step < sample_steps; ++step) {
                float t = sample_times[step];
                float t_next = sample_times[step + 1];
                float t_model = t;
                array z_model = z;
                if (step < sample_steps - 1) {
                    float alpha = std::clamp(
                        1.0f - sde_gamma * (t_next - t), 0.0f, 1.0f);
                    t_model = alpha * t;
                    array eps = multiply(
                        random::normal(z.shape()),
                        array(elf_denoiser_noise_scale, float32));
                    z_model = add(multiply(array(alpha, float16), z),
                                  multiply(array(1.0f - alpha, float16),
                                           astype(eps, float16)));
                    z_model = concatenate({
                        clean_prefix,
                        slice(z_model, {prefix_length, 0},
                              {prefix_length + output_tokens, elf_latent_dim})}, 0);
                }
                array t_array = array(&t_model, {1}, float32);
                x_pred = model.denoise_embeddings(
                    z_model, t_array, stop_gradient(x_pred), std::nullopt,
                    nullptr, cfg_scale_array);
                x_pred = concatenate({
                    clean_prefix,
                    slice(x_pred, {prefix_length, 0},
                          {prefix_length + output_tokens, elf_latent_dim})}, 0);
                float denom = std::max(1.0f - t_model, 0.05f);
                array velocity = divide(subtract(x_pred, z_model),
                                        array(denom, float16));
                z = add(z_model, multiply(
                    array(t_next - t_model, float16), velocity));
                z = concatenate({
                    clean_prefix,
                    slice(z, {prefix_length, 0},
                          {prefix_length + output_tokens, elf_latent_dim})}, 0);
                eval({z, x_pred});
            }
            // The endpoint decoder is trained behind the explicit mode=1
            // condition.  Sampling above follows the denoiser path (mode=0),
            // so run the decoder head once with zero self-conditioning.
            std::optional<array> endpoint_logits;
            if (direct_t < 0.0f) {
                float endpoint_t = 1.0f;
                float decoder_mode_value = 1.0f;
                array endpoint_time = array(&endpoint_t, {1}, float32);
                array decoder_mode = array(&decoder_mode_value, {1}, float32);
                x_pred = model.denoise_embeddings(
                    z, endpoint_time, std::nullopt, decoder_mode,
                    &endpoint_logits, cfg_scale_array);
                eval({x_pred});
            }
            array suffix_pred = slice(
                x_pred, {prefix_length, 0},
                {prefix_length + output_tokens, elf_latent_dim});
            array logits = endpoint_logits
                ? slice(*endpoint_logits, {prefix_length, 0},
                        {prefix_length + output_tokens, generation_vocab})
                : model.decode_embeddings(suffix_pred);
            eval({logits});
            if (t5_latent_generation) {
                array predicted = argmax(logits, -1);
                eval({predicted});
                const uint32_t* predicted_ids = predicted.data<uint32_t>();
                int valid = 0;
                int correct = 0;
                std::cout << "T5_PRED_IDS:";
                for (int i = 0; i < output_tokens; ++i) {
                    int pred = static_cast<int>(predicted_ids[i]);
                    int target = t5_ids[prefix_length + i];
                    std::cout << (i ? "," : "") << pred;
                    if (target != static_cast<int>(elf_t5_dataset->pad_token_id)) {
                        ++valid;
                        if (pred == target) ++correct;
                    }
                }
                std::cout << "\nT5_REF_IDS:";
                for (int i = 0; i < output_tokens; ++i)
                    std::cout << (i ? "," : "") << t5_ids[prefix_length + i];
                std::cout << "\nT5_TOKEN_ACCURACY:"
                          << (valid ? static_cast<double>(correct) / valid : 0.0)
                          << "\nEND\n" << std::flush;
                return 0;
            }
            for (int i = 0; i < output_tokens; ++i) {
                array row = reshape(slice(
                    logits, {i, 0},
                    {i + 1, generation_vocab}), {-1});
                int next = sample_from_probs(row, temp);
                if (next == bpe::EOS_TOKEN_ID) break;
                std::cout << "TOKEN:"
                          << tokenizer.decode(
                                 {static_cast<bpe::TokenId>(next)}, true)
                          << "\n" << std::flush;
            }
            std::cout << "END\n" << std::flush;
            return 0;
        }

        int last_token = -1;
        int repeat_count = 0;
        RecurrentKVCache generation_cache(num_wide_blocks, max_loop);
        std::optional<array> cached_logits;
        if (use_kv_cache) {
            for (TokenId token : ids) {
                int token_value = static_cast<int>(token);
                array token_array = array(&token_value, {1}, int32);
                cached_logits = model.decode(token_array, generation_cache, context_length);
            }
        }

        for (int i = 0; i < max_tokens; ++i) {
            // Training only saw seq_len tokens and CapacityMoE was sized for
            // that length. Bounding the rolling context prevents generation
            // cost and graph size from growing without limit while keeping the
            // default inference distribution aligned with training.
            size_t context_start = ids.size() > static_cast<size_t>(context_length)
                ? ids.size() - static_cast<size_t>(context_length) : 0;
            std::vector<TokenId> context(ids.begin() + context_start, ids.end());
            int current_len = static_cast<int>(context.size());
            array input = array(context.data(), {current_len}, int32);
            
            array logits = use_kv_cache ? *cached_logits : model(input);
            int logits_row = use_kv_cache ? 0 : current_len - 1;
            array last_logit = slice(logits, {logits_row, 0},
                                     {logits_row + 1, static_cast<int>(tokenizer.vocab_size())});
            last_logit = reshape(last_logit, {-1});

            float dynamic_temp = temp;
            if (repeat_count > 1) dynamic_temp = temp * 1.5f;

            int next = sample_from_probs(last_logit, dynamic_temp);
            if (next == bpe::EOS_TOKEN_ID || next == FILE_END_TOKEN_ID) break;

            if (next == last_token) repeat_count++;
            else { last_token = next; repeat_count = 0; }

            ids.push_back(next);
            std::string piece = tokenizer.decode({static_cast<bpe::TokenId>(next)}, true);
            std::cout << "TOKEN:" << piece << "\n" << std::flush;

            if (use_kv_cache) {
                array next_array = array(&next, {1}, int32);
                cached_logits = model.decode(next_array, generation_cache, context_length);
            }
            
            if (gen_clear_interval > 0 && (i + 1) % gen_clear_interval == 0)
                mlx::core::clear_cache();
        }
        std::cout << "END\n" << std::flush;
        return 0;
    }

    // ---------- 数据准备与流式降存 ----------
    packed_batch_size = use_packed_batch ? accum : 1;
    const std::string data_dir = get_arg("--data-dir", "train_files");
    const std::string ckpt_path = get_arg(
        "--model-out", elf_mode ? "openmythos_elf7.ckpt" : "openmythos_mlx.ckpt");
    const std::string tok_path_arg  = get_arg("--tokenizer-out", "tokenizer.bpe");
    if (get_int_arg("--steps", 0) > 0) total_steps = get_int_arg("--steps", total_steps);
    
    TrainingCorpus corpus;
    if (!elf_t5_dataset) corpus = read_training_corpus(data_dir);
    auto &all_texts = corpus.texts;

    // ---------- 分词器处理 ----------
    BPETrainer tokenizer;
    if (!elf_t5_dataset && !tokenizer.load(tok_path_arg)) {
        if (std::filesystem::exists(tok_path_arg))
            throw std::runtime_error("Existing tokenizer is invalid or incompatible; use a new --tokenizer-out path");
        std::cout << "未找到 " << tok_path_arg << "，自动启动高速无锁训练..." << std::endl;
        BPEConfig config;
        const int requested_vocab = get_int_arg("--vocab-size", 16384);
        if (requested_vocab < static_cast<int>(tokenizer.vocab_size())) throw std::invalid_argument("--vocab-size is below the MLX initial vocabulary");
        config.vocab_size = static_cast<size_t>(requested_vocab);
        config.min_frequency = 2;
        tokenizer = BPETrainer(config);

        std::vector<std::string> specials = {
            "[UNK]", "[PAD]", "[BOS]", "[EOS]",
            "\n", "    ", "<THINK>", "</THINK>",          
            "<VERIFY_PASSED>", "<CORRECT>", 
            "<CLASS_DECL:", "<STRUCT_DECL:", "<FUNCTION_DECL:",
            "<FIELD_DECL:", "<VAR_DECL:", "<PARM_DECL:", ">",
            "type=\"", "template"
        };
        if (all_texts.size() > 5000) {
            std::vector<std::string> sample(all_texts.begin(), all_texts.begin() + 5000);
            tokenizer.train_from_texts(sample);
        } else {
            tokenizer.train_from_texts(all_texts);
        }
        for (const auto &token : specials) tokenizer.add_special_token(token);
        if (!tokenizer.save(tok_path_arg, corpus.identity)) throw std::runtime_error("Cannot save MLX tokenizer");
    } else if (!elf_t5_dataset && !tokenizer.dataset_id().empty() && tokenizer.dataset_id() != corpus.identity && get_int_arg("--allow-dataset-change", 0) == 0) {
        throw std::runtime_error("Tokenizer dataset identity mismatch; use a new tokenizer/output path or --allow-dataset-change 1 to reuse the vocabulary intentionally");
    }

    // ---------- 多线程抢单预分词 ----------
    std::cout << "正在多线程预分词..." << std::endl;
    std::vector<std::vector<int>> tokenized_datasets(all_texts.size());
    std::mutex print_mutex;
    unsigned int num_threads = std::min(static_cast<unsigned int>(all_texts.size()),
                                       std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    
    std::atomic<size_t> current_task{0};
    std::atomic<size_t> processed{0};
    const size_t total = all_texts.size();
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                size_t i = current_task.fetch_add(1);
                if (i >= total) break; 
                
                const auto& code = all_texts[i];
                if (!code.empty()) {
                    auto file_tokens = tokenizer.encode(code);
                    std::vector<int> full_tokens;
                    full_tokens.reserve(2 + file_tokens.size());
                    full_tokens.push_back(FILE_START_TOKEN_ID);
                    for (auto id : file_tokens) full_tokens.push_back(static_cast<int>(id));
                    full_tokens.push_back(FILE_END_TOKEN_ID);
                    tokenized_datasets[i] = std::move(full_tokens);
                }
    
                size_t done = processed.fetch_add(1) + 1;
                if (done % 500 == 0 || done == total) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "\r  预分词进度: " << done << "/" << total << " (" << (done * 100 / total) << "%)" << std::flush;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    
    tokenized_datasets.erase(
        std::remove_if(tokenized_datasets.begin(), tokenized_datasets.end(),
                       [](const auto& v) { return v.empty(); }),
        tokenized_datasets.end());
    std::cout << "\n✅ 预分词完成，有效序列数: " << tokenized_datasets.size() << std::endl;
    all_texts.clear();
    if (!elf_t5_dataset && tokenized_datasets.empty()) throw std::invalid_argument("No nonempty tokenized training sequences");

    if (elf_t5_dataset && elf_t5_dataset->seq_len != static_cast<uint32_t>(seq_len))
        throw std::invalid_argument("T5 latent sequence length does not match --seq-len");
    const int training_vocab_size = elf_t5_dataset
        ? static_cast<int>(elf_t5_dataset->vocab_size)
        : (elf_t5_unembedding
            ? static_cast<int>(elf_t5_unembedding->vocab_size)
            : static_cast<int>(tokenizer.vocab_size()));
    if (elf_t5_dataset && elf_t5_unembedding &&
        elf_t5_dataset->vocab_size != elf_t5_unembedding->vocab_size)
        throw std::invalid_argument("T5 latent/unembedding vocabulary mismatch");
    if (elf_t5_dataset) {
        std::cout << "🧬 T5 latent dataset: records=" << elf_t5_dataset->records
                  << ", seq=" << elf_t5_dataset->seq_len
                  << ", latent_dim=" << elf_t5_dataset->latent_dim
                  << ", vocab=" << elf_t5_dataset->vocab_size << "\n";
    }

    // ---------- 模型与底层内存池初始化 ----------
#ifdef USE_CAPACITY_MOE
    // 路由噪声强度：通过 --router-noise-std 命令行参数调整，默认0（不开启）。
    // 这是 Noisy Top-K Gating 机制——训练时给路由logits加高斯噪声，打破
    // "赢家通吃"的专家坍缩循环，和负载均衡辅助损失（aux_loss_weight）是
    // 互补关系，建议两者一起用。常见起始值在0.1~1.0之间，标准差越大探索性
    // 越强，但太大会让路由判断变得接近随机，需要实测调整。
    const float router_noise_std = parse_finite_float(get_arg("--router-noise-std", "0.0"));
    if (router_noise_std > 0.0f && use_compiled_train_graph) {
        std::cout << "⚠️ 路由噪声启用时自动关闭整图编译，避免缓存隐式随机状态。\n";
        use_compiled_train_graph = false;
    }
    std::cout << "📐 路由噪声标准差 (router_noise_std) = " << router_noise_std
              << (router_noise_std > 0.0f ? "" : " (未启用)") << std::endl;
    OpenMythos model(training_vocab_size, dim, max_loop, window_size, global_topk,
                     moe_experts, moe_topk, num_wide_blocks, seq_len, 1.4f,
                     router_noise_std, elf_mode, elf_latent_dim);
#else
    OpenMythos model(training_vocab_size, dim, max_loop, window_size, global_topk,
                     moe_experts, moe_topk, num_wide_blocks, seq_len,
                     1.4f, 0.0f, elf_mode, elf_latent_dim);
#endif
    if (elf_t5_unembedding)
        model.set_elf_unembedding(elf_t5_unembedding->weights);
    if (elf_t5_dataset)
        model.set_elf_latent_normalization(elf_latent_mean, elf_latent_std);
    
    apply_galore_cfg(model.mp);
    if (galore_off) {
        std::cout << "📐 GaLore 已通过 --galore-off 关闭，使用原版全量优化器" << std::endl;
    }
    model.mp.init_swap();

    // 🖥️ 终端监控面板的滚动历史buffer：保留最近60个采样点喂给sparkline，
    // 实际显示宽度由 render_sparkline 内部降采样到固定30个字符，
    // 不会随历史数据量增长而越打越长。这是"最近细节"视图。
    RollingHistory loss_history(60);
    RollingHistory gnorm_history(60);
#ifdef USE_CAPACITY_MOE
    RollingHistory aux_history(60);
#endif

    // 全程历史（不设容量上限，从训练第一步记到当前），用于展示"整体趋势"。
    // 同样喂给 render_sparkline，内部的 downsample 会自动压缩到固定宽度，
    // 不管训练跑了100步还是5000步，这条线显示出来的宽度都是一样的——
    // 只是数据点越多，每个显示字符代表的"时间跨度"就越长（分桶平均）。
    // 5000步顶多存5000个float（约20KB），内存开销可忽略。
    std::vector<float> loss_history_full;
    loss_history_full.reserve(total_steps);
#ifdef USE_CAPACITY_MOE
    std::vector<float> aux_history_full;
    aux_history_full.reserve(total_steps);
#endif

    // 用于在面板上标注"训练起点loss"作为参照，方便直观对比现在比起点下降了多少
    float first_loss_recorded = -1.0f;

    // 专家负载面板只在每200步更新一次，但每次刷新整屏时都要带着一起打印，
    // 否则清屏后这部分内容会在中间几次10步刷新时"消失"。这里用一个常驻
    // 字符串存最近一次的采样结果，平时只是重复显示，不重新计算。
    std::string last_expert_panel = "  (尚无专家负载数据，将在首次采样时显示)\n";

#ifdef USE_CAPACITY_MOE
    // 不均衡程度的历史趋势：每次采样专家负载时，记录"最高负载/理论均衡负载"
    // 这个比值（1.0=完美均衡，数值越大越不均衡）。单次快照容易受随机噪声
    // 影响，看不出训练推进过程中负载均衡损失是否真的在起效——这条趋势线
    // 能回答"不均衡程度是不是在缓慢改善，还是一直卡在某个水平没动"。
    std::vector<float> imbalance_ratio_history;
    imbalance_ratio_history.reserve(total_steps / 10 + 1);  // 粗略预估采样次数
#endif

    const auto model_metadata = metadata_for(tokenizer, training_vocab_size);
    int start_step = 0;
    {
        std::ifstream is(ckpt_path, std::ios::binary);
        if (is) {
            mlx_checkpoint::read_metadata(is, model_metadata);
            is.read(reinterpret_cast<char*>(&start_step), sizeof(int));
            if (!is || start_step < 0) throw std::runtime_error("Invalid checkpoint training step");
            if (elf_mode) {
                uint32_t magic = 0;
                is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                if (magic != elf_checkpoint_magic) {
                    throw std::runtime_error("Checkpoint ELF layout mismatch: use a new ELF7 output path; older checkpoints require explicit migration.");
                }
            }
            if (is && model.mp.load(is)) std::cout << "从 Checkpoint (Step: " << start_step << ") 成功恢复状态。" << std::endl;
            else throw std::runtime_error("Failed to restore checkpoint");
        }
    }

    // ---------- 训练主循环 ----------
    array inp = zeros({seq_len}, int32);
    array tgt = zeros({seq_len}, int32);

#ifdef USE_CAPACITY_MOE
    // 负载均衡辅助损失权重：通过 --aux-weight 命令行参数调整，不需要改代码
    // 重新编译就能试不同的值。
    //
    // 默认值从最初的0.01上调到0.05——实测数据显示：0.01权重下，aux_loss贡献
    // 只占总loss的约2%（main_loss≈3.45, aux_raw≈7.25, 0.01*7.25=0.0725），
    // 这个力度明显压不过主任务loss的优化压力，导致专家坍缩没能被有效抑制
    // （400步时观察到E4/E6两个专家占了261/384=68%的token，capacity都溢出了）。
    // 0.05时占比约9.5%，给负载均衡更实质性的约束力，但又不至于压垮主任务学习。
    // 如果0.05还不够，可以试0.1（占比~17%）；如果主任务loss下降明显变慢，
    // 说明矫枉过正了，往回调小。
    const float aux_loss_weight = parse_finite_float(get_arg("--aux-weight", "0.05"));
    std::cout << "📐 负载均衡辅助损失权重 (aux_loss_weight) = " << aux_loss_weight << std::endl;

    // 专家负载监控采样间隔：通过 --expert-monitor-interval 调整，默认每200步
    // 采一次。调小这个值能更密集地观察负载分布变化，但代价是每次采样都要
    // 多跑一次 router(x)（相对FFN计算量很小，频率不要调得过分密集即可，
    // 比如每10步采一次问题不大，但没必要每步都采）。
    const int expert_monitor_interval = get_int_arg("--expert-monitor-interval", 200);
    if (expert_monitor_interval <= 0) throw std::invalid_argument("--expert-monitor-interval must be positive");
    std::cout << "📐 专家负载监控采样间隔 = 每" << expert_monitor_interval << "步" << std::endl;
#endif

    if (elf_mode) {
        const int elf_batches = use_packed_batch ? accum : 1;
        const int elf_prefix = get_int_arg("--elf-prefix-tokens", seq_len / 2);
        const float elf_decoder_weight = parse_finite_float(
            get_arg("--elf-decoder-weight", "0.1"));
        const float elf_decoder_full_weight = parse_finite_float(
            get_arg("--elf-decoder-full-weight", "1.0"));
        const float elf_decoder_branch_prob = parse_finite_float(
            get_arg("--elf-decoder-branch-prob", "0.2"));
        const float elf_decoder_noise_scale = parse_finite_float(
            get_arg("--elf-decoder-noise-scale", elf_t5_dataset ? "5.0" : "1.0"));
        const float elf_decoder_p_mean = parse_finite_float(
            get_arg("--elf-decoder-p-mean", "0.8"));
        const float elf_decoder_p_std = parse_finite_float(
            get_arg("--elf-decoder-p-std", "0.8"));
        const bool elf_exclusive_decoder =
            get_int_arg("--elf-exclusive-decoder", 1) != 0;
        const float elf_context_mix = parse_finite_float(
            get_arg("--elf-context-mix", "0.15"));
        const float elf_self_cond_prob = parse_finite_float(
            get_arg("--elf-self-cond-prob", "0.5"));
        const float elf_self_cond_cfg_min = parse_finite_float(
            get_arg("--elf-self-cond-cfg-min", "0.5"));
        const float elf_self_cond_cfg_max = parse_finite_float(
            get_arg("--elf-self-cond-cfg-max", "5.0"));
        const float elf_ema_decay = parse_finite_float(
            get_arg("--elf-ema-decay", "0.9999"));
        const bool elf_x0_loss = get_int_arg(
            "--elf-x0-loss", 0) != 0;
        const float elf_time_min = parse_finite_float(get_arg(
            "--elf-time-min", elf_x0_loss ? "0.0" : "0.05"));
        const float elf_time_max = parse_finite_float(get_arg("--elf-time-max", "0.95"));
        const std::string elf_time_schedule = get_arg(
            "--elf-time-schedule", elf_t5_dataset ? "logit_normal" : "uniform");
        const float elf_time_p_mean = parse_finite_float(
            get_arg("--elf-time-p-mean", elf_t5_dataset ? "-1.5" : "0.8"));
        const float elf_time_p_std = parse_finite_float(
            get_arg("--elf-time-p-std", "0.8"));
        if (elf_prefix < 0 || elf_prefix >= seq_len)
            throw std::invalid_argument("Invalid --elf-prefix-tokens");
        if (elf_context_mix < 0.0f || elf_context_mix >= 0.5f ||
            elf_self_cond_prob < 0.0f || elf_self_cond_prob > 1.0f ||
            elf_self_cond_cfg_min < 0.0f ||
            elf_self_cond_cfg_max < elf_self_cond_cfg_min ||
            elf_decoder_weight < 0.0f || elf_decoder_full_weight < 0.0f ||
            elf_decoder_branch_prob < 0.0f || elf_decoder_branch_prob > 1.0f ||
            elf_decoder_noise_scale < 0.0f || elf_decoder_p_std <= 0.0f ||
            elf_ema_decay < 0.0f || elf_ema_decay >= 1.0f ||
            elf_time_min < 0.0f || elf_time_max > 1.0f ||
            elf_time_min >= elf_time_max || elf_time_p_std <= 0.0f ||
            (elf_time_schedule != "uniform" && elf_time_schedule != "logit_normal"))
            throw std::invalid_argument("Invalid ELF2 context/self-conditioning configuration");
        std::cout << "🧬 ELF mode: batch=" << elf_batches
                  << ", prefix=" << elf_prefix
                  << ", decoder_weight=" << elf_decoder_weight
                  << ", decoder_full_weight=" << elf_decoder_full_weight
                  << ", decoder_branch_prob=" << elf_decoder_branch_prob
                  << ", decoder_noise_scale=" << elf_decoder_noise_scale
                  << ", decoder_p_mean=" << elf_decoder_p_mean
                  << ", decoder_p_std=" << elf_decoder_p_std
                  << ", exclusive_decoder=" << elf_exclusive_decoder
                  << ", objective=" << (elf_x0_loss ? "x0" : "velocity")
                  << ", time_schedule=" << elf_time_schedule
                  << ", time_p_mean=" << elf_time_p_mean
                  << ", time_p_std=" << elf_time_p_std
                  << ", time_range=[" << elf_time_min << "," << elf_time_max << "]"
                  << ", latent_mean=" << elf_latent_mean
                  << ", latent_std=" << elf_latent_std
                  << ", denoiser_noise_scale=" << elf_denoiser_noise_scale
                  << ", context_mix=" << elf_context_mix
                  << ", self_cond_prob=" << elf_self_cond_prob
                  << ", self_cond_cfg=[" << elf_self_cond_cfg_min
                  << "," << elf_self_cond_cfg_max << "]\n";
        std::cout << "🧬 ELF EMA decay=" << elf_ema_decay
                  << ", inference checkpoint=" << ckpt_path << ".ema\n";

        const size_t elf_param_count = model.mp.ptrs.size();
        std::vector<int> elf_argnums(elf_param_count);
        std::iota(elf_argnums.begin(), elf_argnums.end(), 0);
        auto elf_forward = [&](const std::vector<array>& inputs)
            -> std::vector<array> {
            std::vector<array> params(
                inputs.begin(), inputs.begin() + elf_param_count);
            model.mp.set_values(params);
            const array& ids = inputs[elf_param_count];
            const array& noise = inputs[elf_param_count + 1];
            const array& times = inputs[elf_param_count + 2];
            const array& self_cond_mask = inputs[elf_param_count + 3];
            const array& decoder_branch_mask = inputs[elf_param_count + 4];
            const array& external_latents = inputs[elf_param_count + 5];
            const array& decoder_lambda_noise = inputs[elf_param_count + 6];
            const array& decoder_noise_base = inputs[elf_param_count + 7];
            const array& cfg_scales = inputs[elf_param_count + 8];
            const array& initial_predictions = inputs[elf_param_count + 9];
            const array& guided_v_targets = inputs[elf_param_count + 10];
            int length = ids.shape(0);
            array x0 = elf_t5_dataset
                ? stop_gradient(divide(
                    subtract(astype(external_latents, float32),
                             array(elf_latent_mean, float32)),
                    array(elf_latent_std, float32)))
                : stop_gradient(model.contextual_token_embeddings(ids, elf_context_mix));
            int latent_width = x0.shape(1);
            array tr = reshape(broadcast_to(
                reshape(times, {elf_batches, 1, 1}),
                {elf_batches, seq_len, latent_width}), {length, latent_width});
            array z = add(multiply(tr, x0),
                          multiply(subtract(array(1.0f, float32), tr), noise));

            array positions = reshape(broadcast_to(
                expand_dims(arange(seq_len), 0),
                {elf_batches, seq_len}), {length});
            array prefix_mask = less(positions, array(elf_prefix, int32));
            z = where(expand_dims(prefix_mask, -1), x0, z);
            array decoder_lambda = sigmoid(add(
                array(elf_decoder_p_mean, float32),
                multiply(array(elf_decoder_p_std, float32),
                         decoder_lambda_noise)));
            array decoder_noise = multiply(
                decoder_noise_base,
                array(elf_decoder_noise_scale, float32));
            array decoder_z = add(
                multiply(decoder_lambda, x0),
                multiply(subtract(array(1.0f, float32), decoder_lambda), decoder_noise));
            array decoder_active = greater(
                decoder_branch_mask, array(0.5f, float32));
            z = where(expand_dims(prefix_mask, -1), x0,
                      where(decoder_active, decoder_z, z));
            array z_f16 = astype(z, float16);
            array mode_batch = broadcast_to(
                reshape(decoder_branch_mask, {1}), {elf_batches});
            array prefix_2d = expand_dims(prefix_mask, -1);
            array initial_pred = where(
                prefix_2d, x0, astype(initial_predictions, float32));
            array sc = reshape(broadcast_to(
                reshape(self_cond_mask, {elf_batches, 1, 1}),
                {elf_batches, seq_len, latent_width}), {length, latent_width});
            array self_condition = multiply(initial_pred, sc);
            self_condition = where(prefix_2d, x0, self_condition);
            self_condition = where(decoder_active, zeros_like(self_condition), self_condition);
            std::optional<array> decoder_logits;
            array pred = model.denoise_embeddings(
                z_f16, times, astype(self_condition, float16), mode_batch,
                &decoder_logits, cfg_scales);

            array denom = maximum(
                subtract(array(1.0f, float32), tr), array(0.05f, float32));
            array v_pred = divide(subtract(astype(pred, float32), z), denom);
            array v_target = guided_v_targets;
            array valid = logical_and(
                logical_not(prefix_mask),
                not_equal(ids, array(elf_t5_dataset
                    ? static_cast<int>(elf_t5_dataset->pad_token_id)
                    : static_cast<int>(PAD_TOKEN_ID), int32)));
            array valid_f = astype(valid, float32);
            array flow_per_token = mean(square(subtract(v_pred, v_target)), -1);
            if (elf_x0_loss)
                flow_per_token = mean(square(
                    subtract(astype(pred, float32), x0)), -1);
            array token_count = maximum(sum(valid_f), array(1.0f, float32));
            array flow_loss = divide(
                sum(multiply(flow_per_token, valid_f)), token_count);

            array logits = astype(*decoder_logits, float32);
            array shifted = subtract(logits, max(logits, -1, true));
            array log_probs = subtract(
                shifted, log(sum(exp(shifted), -1, true)));
            array target_logp = reshape(take_along_axis(
                log_probs, expand_dims(ids, -1), -1), {-1});
            array decoder_loss = divide(
                sum(multiply(negative(target_logp), valid_f)), token_count);
            array decoder_weight = where(
                greater(decoder_branch_mask, array(0.5f, float32)),
                array(elf_decoder_full_weight, float32),
                array(elf_decoder_weight, float32));
#ifdef USE_CAPACITY_MOE
            array aux = model.total_aux_loss();
            array task_loss = elf_exclusive_decoder
                ? where(greater(decoder_branch_mask, array(0.5f, float32)),
                        multiply(decoder_weight, decoder_loss), flow_loss)
                : add(flow_loss, multiply(decoder_weight, decoder_loss));
            array total = add(task_loss,
                multiply(array(aux_loss_weight, float32), aux));
            return {total, flow_loss, decoder_loss, aux};
#else
            array total = elf_exclusive_decoder
                ? where(greater(decoder_branch_mask, array(0.5f, float32)),
                        multiply(decoder_weight, decoder_loss), flow_loss)
                : add(flow_loss, multiply(decoder_weight, decoder_loss));
            return {total, flow_loss, decoder_loss};
#endif
        };
        auto elf_value_grad = value_and_grad(elf_forward, elf_argnums);
        auto elf_graph = [&](const std::vector<array>& inputs) {
            auto [values, grads] = elf_value_grad(inputs);
            values.insert(values.end(), grads.begin(), grads.end());
            return values;
        };
        auto compiled_elf_graph = compile(
            std::function<std::vector<array>(const std::vector<array>&)>(elf_graph));
        std::vector<array> elf_ema_params = model.mp.get_values();
        eval(elf_ema_params);
        // Resume the shadow weights as well as the live model.  Loading the EMA
        // checkpoint temporarily also loads its optimizer buffers, so restore the
        // main checkpoint immediately afterwards before taking another step.
        if (start_step > 0) {
            const std::string ema_path = ckpt_path + ".ema";
            std::ifstream ema_is(ema_path, std::ios::binary);
            if (ema_is) {
                mlx_checkpoint::read_metadata(ema_is, model_metadata);
                int ema_step = 0;
                uint32_t ema_magic = 0;
                ema_is.read(reinterpret_cast<char*>(&ema_step), sizeof(ema_step));
                ema_is.read(reinterpret_cast<char*>(&ema_magic), sizeof(ema_magic));
                if (ema_is && ema_step == start_step &&
                    ema_magic == elf_checkpoint_magic && model.mp.load(ema_is)) {
                    elf_ema_params = model.mp.get_values();
                    eval(elf_ema_params);
                    std::cout << "ELF EMA restored from step " << ema_step << ".\n";

                    std::ifstream live_is(ckpt_path, std::ios::binary);
                    mlx_checkpoint::read_metadata(live_is, model_metadata);
                    int live_step = 0;
                    uint32_t live_magic = 0;
                    live_is.read(reinterpret_cast<char*>(&live_step), sizeof(live_step));
                    live_is.read(reinterpret_cast<char*>(&live_magic), sizeof(live_magic));
                    if (!live_is || live_step != start_step ||
                        live_magic != elf_checkpoint_magic || !model.mp.load(live_is))
                        throw std::runtime_error(
                            "Failed to restore live checkpoint after loading EMA");
                } else {
                    std::cerr << "ELF EMA checkpoint does not match live step; "
                                 "reinitializing EMA from live weights.\n";
                }
            }
        }
        auto save_elf_checkpoint = [&](const std::string& path, int step,
                                       const std::vector<array>* values) {
            std::vector<array> live_params;
            if (values) {
                live_params = model.mp.get_values();
                model.mp.set_values(*values);
            }
            try {
                mlx_checkpoint::save_atomic(path, model_metadata, [&](std::ostream &os) {
                    os.write(reinterpret_cast<const char*>(&step), sizeof(step));
                    os.write(reinterpret_cast<const char*>(&elf_checkpoint_magic), sizeof(elf_checkpoint_magic));
                    model.mp.save(os);
                });
            } catch (...) {
                if (values) model.mp.set_values(live_params);
                throw;
            }
            if (values) model.mp.set_values(live_params);
            return true;
        };
        std::uniform_real_distribution<float> time_dist(
            elf_time_min, elf_time_max);
        std::normal_distribution<float> time_normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> cfg_uniform(0.0f, 1.0f);
        auto sample_elf_time = [&]() {
            if (elf_time_schedule == "logit_normal") {
                float value = elf_time_p_mean + elf_time_p_std * time_normal(rng);
                return 1.0f / (1.0f + std::exp(-value));
            }
            return time_dist(rng);
        };

        for (int s = start_step; s < total_steps; ++s) {
            const int elf_pad_id = elf_t5_dataset
                ? static_cast<int>(elf_t5_dataset->pad_token_id)
                : static_cast<int>(PAD_TOKEN_ID);
            std::vector<int32_t> ids_host(elf_batches * seq_len, elf_pad_id);
            std::vector<float16_t> latent_host(
                static_cast<size_t>(elf_batches) * seq_len * elf_latent_dim);
            for (int b = 0; b < elf_batches; ++b) {
                if (elf_t5_dataset) {
                    uint32_t record = rng() % elf_t5_dataset->records;
                    elf_t5_dataset->read_record(
                        record, ids_host.data() + b * seq_len,
                        latent_host.data() + static_cast<size_t>(b) * seq_len * elf_latent_dim);
                } else {
                    const auto& tokens = tokenized_datasets[
                        rng() % tokenized_datasets.size()];
                    int start = 0;
                    if (tokens.size() > static_cast<size_t>(seq_len))
                        start = rng() % (tokens.size() - seq_len + 1);
                    int count = std::min(seq_len, (int)tokens.size() - start);
                    std::copy(tokens.begin() + start, tokens.begin() + start + count,
                              ids_host.begin() + b * seq_len);
                }
            }
            std::bernoulli_distribution decoder_branch_dist(elf_decoder_branch_prob);
            float decoder_branch_host = decoder_branch_dist(rng) ? 1.0f : 0.0f;
            std::vector<float> times_host(elf_batches);
            for (float& t : times_host)
                t = (elf_exclusive_decoder && decoder_branch_host > 0.5f)
                    ? 1.0f : sample_elf_time();
            std::bernoulli_distribution self_cond_dist(elf_self_cond_prob);
            std::vector<float> self_cond_host(elf_batches);
            for (float& enabled : self_cond_host)
                enabled = self_cond_dist(rng) ? 1.0f : 0.0f;
            std::vector<float> cfg_scale_host(elf_batches);
            const float cfg_log_ratio = std::log(
                (1.0f + elf_self_cond_cfg_max) /
                (1.0f + elf_self_cond_cfg_min));
            for (float& scale : cfg_scale_host) {
                float u = cfg_uniform(rng);
                scale = (1.0f + elf_self_cond_cfg_min) *
                    std::exp(u * cfg_log_ratio) - 1.0f;
            }
            array ids = array(ids_host.data(), {elf_batches * seq_len}, int32);
            array noise = divide(
                multiply(random::normal({elf_batches * seq_len, elf_latent_dim}),
                         array(elf_denoiser_noise_scale, float32)),
                array(1.0f, float32));
            array times = array(times_host.data(), {elf_batches}, float32);
            array self_cond_mask = array(
                self_cond_host.data(), {elf_batches}, float32);
            array cfg_scales = array(
                cfg_scale_host.data(), {elf_batches}, float32);
            array decoder_branch_mask = array(decoder_branch_host, float32);
            array latent_targets = array(
                latent_host.data(), {elf_batches * seq_len, elf_latent_dim}, float16);
            // Keep randomness outside the compiled function. MLX compile captures
            // an implicit PRNG key during tracing, which otherwise replays the
            // same decoder corruption on every invocation.
            array decoder_lambda_noise = random::normal(
                {elf_batches * seq_len, 1});
            array decoder_noise_base = random::normal(
                {elf_batches * seq_len, elf_latent_dim});
            array initial_predictions = zeros(
                {elf_batches * seq_len, elf_latent_dim}, float16);
            array guided_v_targets = zeros(
                {elf_batches * seq_len, elf_latent_dim}, float32);
            if (decoder_branch_host <= 0.5f) {
                array x0_step = elf_t5_dataset
                    ? stop_gradient(divide(
                        subtract(astype(latent_targets, float32),
                                 array(elf_latent_mean, float32)),
                        array(elf_latent_std, float32)))
                    : stop_gradient(astype(
                        model.contextual_token_embeddings(ids, elf_context_mix),
                        float32));
                int step_length = elf_batches * seq_len;
                array step_t = reshape(broadcast_to(
                    reshape(times, {elf_batches, 1, 1}),
                    {elf_batches, seq_len, elf_latent_dim}),
                    {step_length, elf_latent_dim});
                array z_step = add(
                    multiply(step_t, x0_step),
                    multiply(subtract(array(1.0f, float32), step_t), noise));
                array step_positions = reshape(broadcast_to(
                    expand_dims(arange(seq_len), 0),
                    {elf_batches, seq_len}), {step_length});
                array step_prefix = expand_dims(
                    less(step_positions, array(elf_prefix, int32)), -1);
                z_step = where(step_prefix, x0_step, z_step);
                array uncond_self = where(
                    step_prefix, x0_step, zeros_like(x0_step));
                array denoiser_mode = zeros({elf_batches}, float32);
                initial_predictions = stop_gradient(model.denoise_embeddings(
                    astype(z_step, float16), times,
                    astype(uncond_self, float16), denoiser_mode, nullptr,
                    cfg_scales));
                initial_predictions = stop_gradient(where(
                    step_prefix, astype(x0_step, float16),
                    initial_predictions));
                array conditional_pred = stop_gradient(model.denoise_embeddings(
                    astype(z_step, float16), times, initial_predictions,
                    denoiser_mode, nullptr, cfg_scales));
                conditional_pred = stop_gradient(where(
                    step_prefix, astype(x0_step, float16), conditional_pred));
                array step_denom = maximum(
                    subtract(array(1.0f, float32), step_t),
                    array(0.05f, float32));
                array base_v = divide(subtract(x0_step, z_step), step_denom);
                array v_uncond = divide(
                    subtract(astype(initial_predictions, float32), z_step),
                    step_denom);
                array v_cond = divide(
                    subtract(astype(conditional_pred, float32), z_step),
                    step_denom);
                array cfg_coeff = reshape(broadcast_to(
                    reshape(subtract(array(1.0f, float32),
                                     divide(array(1.0f, float32), cfg_scales)),
                            {elf_batches, 1, 1}),
                    {elf_batches, seq_len, elf_latent_dim}),
                    {step_length, elf_latent_dim});
                array guide_mask = reshape(broadcast_to(
                    reshape(self_cond_mask, {elf_batches, 1, 1}),
                    {elf_batches, seq_len, elf_latent_dim}),
                    {step_length, elf_latent_dim});
                guided_v_targets = stop_gradient(add(
                    base_v, multiply(guide_mask,
                        multiply(cfg_coeff, subtract(v_cond, v_uncond)))));
                eval({initial_predictions, guided_v_targets});
            }
            std::vector<array> real_params = model.mp.get_values();
            std::vector<array> graph_inputs = real_params;
            graph_inputs.insert(
                graph_inputs.end(), {ids, noise, times, self_cond_mask,
                                     decoder_branch_mask, latent_targets,
                                     decoder_lambda_noise, decoder_noise_base,
                                     cfg_scales, initial_predictions,
                                     guided_v_targets});
            auto outputs = use_compiled_train_graph
                ? compiled_elf_graph(graph_inputs) : elf_graph(graph_inputs);
            model.mp.set_values(real_params);
#ifdef USE_CAPACITY_MOE
            const size_t metric_count = 4;
#else
            const size_t metric_count = 3;
#endif
            std::vector<array> grads(outputs.begin() + metric_count, outputs.end());
            std::vector<array> evaluated = outputs;
            eval(evaluated);

            array global_sq_norm = array(0.0f, float32);
            for (const auto& g : grads)
                global_sq_norm = add(global_sq_norm,
                                     sum(square(astype(g, float32))));
            array gnorm = sqrt(global_sq_norm);
            array scale = where(
                greater(gnorm, array(5.0f, float32)),
                divide(array(5.0f, float32), gnorm),
                array(1.0f, float32));
            for (auto& g : grads) g = multiply(g, astype(scale, float16));
            float lr = get_lr(s, total_steps, peak_lr, 200);
            float b1_corr = 1.0f - std::pow(0.9f, s + 1);
            float b2_corr = 1.0f - std::pow(0.999f, s + 1);
            apply_snr_gated_update(
                model.mp, grads, lr, b1_corr, b2_corr, 9999.0f, s, {gnorm});
            std::vector<array> updated_params = model.mp.get_values();
            for (size_t i = 0; i < elf_ema_params.size(); ++i) {
                elf_ema_params[i] = add(
                    multiply(array(elf_ema_decay, elf_ema_params[i].dtype()),
                             elf_ema_params[i]),
                    multiply(array(1.0f - elf_ema_decay, updated_params[i].dtype()),
                             updated_params[i]));
            }
            eval(elf_ema_params);

            if (s % 10 == 0) {
                std::cout << "ELF Step " << s
                          << " total=" << outputs[0].item<float>()
                          << " flow=" << outputs[1].item<float>()
                          << " decoder=" << outputs[2].item<float>()
                          << " branch=" << (decoder_branch_host > 0.5f
                              ? "decoder" : "denoiser")
#ifdef USE_CAPACITY_MOE
                          << " aux=" << outputs[3].item<float>()
#endif
                          << " grad=" << gnorm.item<float>() << "\n";
            }
            if (clear_interval > 0 && (s + 1) % clear_interval == 0)
                clear_cache();
            if (s > 0 && s % 100 == 0) {
                if (save_elf_checkpoint(ckpt_path, s + 1, nullptr)) {
                    save_elf_checkpoint(ckpt_path + ".ema", s + 1, &elf_ema_params);
                    std::cout << "ELF checkpoint saved at step " << s << "\n";
                }
            }
        }
        if (get_int_arg("--save-final", 1) != 0) {
            int completed_step = total_steps;
            if (save_elf_checkpoint(ckpt_path, completed_step, nullptr)) {
                save_elf_checkpoint(
                    ckpt_path + ".ema", completed_step, &elf_ema_params);
                std::cout << "ELF final checkpoint saved at step "
                          << completed_step << "\n";
            }
        }
        return 0;
    }

    auto loss_fn = [&](const std::vector<array>& params) -> array {
        model.mp.set_values(params);
        array logits = model(inp);
        array main_loss = cross_entropy_loss(logits, tgt, PAD_TOKEN_ID);
#ifdef USE_CAPACITY_MOE
        array aux = model.total_aux_loss();
        return add(main_loss, multiply(array(aux_loss_weight, float32), aux));
#else
        return main_loss;
#endif
    };

    std::vector<int> argnums(model.mp.ptrs.size());
    std::iota(argnums.begin(), argnums.end(), 0);
    auto grad_fn = value_and_grad(loss_fn, argnums);

    const size_t train_param_count = model.mp.ptrs.size();
    auto explicit_loss_fn = [&](const std::vector<array>& inputs) -> std::vector<array> {
        std::vector<array> params(inputs.begin(), inputs.begin() + train_param_count);
        model.mp.set_values(params);
        array logits = model(inputs[train_param_count]);
        array main_loss = cross_entropy_loss(
            logits, inputs[train_param_count + 1], PAD_TOKEN_ID);
#ifdef USE_CAPACITY_MOE
        array aux = model.total_aux_loss();
        return {add(main_loss, multiply(array(aux_loss_weight, float32), aux)), aux};
#else
        return {main_loss};
#endif
    };
    auto explicit_grad_fn = value_and_grad(explicit_loss_fn, argnums);
    auto train_graph_fn = [&](const std::vector<array>& inputs) -> std::vector<array> {
        auto [values, grads] = explicit_grad_fn(inputs);
        std::vector<array> outputs = std::move(values);
        outputs.reserve(outputs.size() + grads.size());
        outputs.insert(outputs.end(), grads.begin(), grads.end());
        return outputs;
    };
    auto compiled_grad_fn = compile(
        std::function<std::vector<array>(const std::vector<array>&)>(train_graph_fn));

    struct TrainingGraphResult {
        array loss;
        array aux;
        std::vector<array> grads;
    };
    auto run_training_graph = [&](const array& input, const array& target) {
        inp = input;
        tgt = target;
        array loss = array(0.0f, float32);
        array micro_aux = array(0.0f, float32);
        std::vector<array> grads;
        if (use_compiled_train_graph) {
            std::vector<array> real_params = model.mp.get_values();
            std::vector<array> graph_inputs = real_params;
            graph_inputs.push_back(inp);
            graph_inputs.push_back(tgt);
            auto graph_outputs = compiled_grad_fn(graph_inputs);
            // Tracing temporarily installs placeholders through set_values().
            // Restore real trainable arrays before the optimizer mutates them.
            model.mp.set_values(real_params);
            loss = graph_outputs[0];
            size_t grad_offset = 1;
#ifdef USE_CAPACITY_MOE
            micro_aux = graph_outputs[1];
            grad_offset = 2;
#endif
            grads.assign(graph_outputs.begin() + grad_offset, graph_outputs.end());
        } else {
            auto legacy_result = grad_fn(model.mp.get_values());
            loss = legacy_result.first;
            grads = std::move(legacy_result.second);
#ifdef USE_CAPACITY_MOE
            micro_aux = model.total_aux_loss();
#endif
        }
        return TrainingGraphResult{loss, micro_aux, std::move(grads)};
    };

    auto sample_window = [&]() {
        const auto& full_tokens = tokenized_datasets[rng() % tokenized_datasets.size()];
        int total_len = full_tokens.size();
        int win_start = 0;
        if (total_len > seq_len + 1) {
            int max_start = total_len - (seq_len + 1);
            win_start = rng() % (max_start + 1);
        }
        int win_end = std::min(win_start + seq_len + 1, total_len);
        int cur_len = win_end - win_start;
        std::vector<int> input(seq_len, PAD_TOKEN_ID);
        std::vector<int> target(seq_len, PAD_TOKEN_ID);
        std::copy(full_tokens.begin() + win_start,
                  full_tokens.begin() + win_start + cur_len - 1, input.begin());
        std::copy(full_tokens.begin() + win_start + 1,
                  full_tokens.begin() + win_start + cur_len, target.begin());
        return std::make_pair(std::move(input), std::move(target));
    };

    for (int s = start_step; s < total_steps; ++s) {
        // 🌟 核心：大幅缩短 Warmup (从 1000 斩断到 200)，让大 LR 瞬间注入权重
        float lr = get_lr(s, total_steps, peak_lr, 200); 
        float b1_corr = 1.0f - std::pow(0.9f, s + 1);
        float b2_corr = 1.0f - std::pow(0.999f, s + 1);

        std::vector<array> accum_grads;
        accum_grads.reserve(model.mp.ptrs.size());
        array step_loss_arr = array(0.0f, float32);
#ifdef USE_CAPACITY_MOE
        // 独立追踪辅助损失（不带权重的原始值），方便在面板上单独显示，
        // 避免和主任务loss混在一起后看不清各自的变化趋势。
        array step_aux_arr = array(0.0f, float32);
#endif

        if (use_packed_batch) {
            std::vector<int> packed_input;
            std::vector<int> packed_target;
            packed_input.reserve(accum * seq_len);
            packed_target.reserve(accum * seq_len);
            for (int a = 0; a < accum; ++a) {
                auto [input, target] = sample_window();
                packed_input.insert(packed_input.end(), input.begin(), input.end());
                packed_target.insert(packed_target.end(), target.begin(), target.end());
            }
            array batch_input = array(packed_input.data(), {accum * seq_len}, int32);
            array batch_target = array(packed_target.data(), {accum * seq_len}, int32);
            inp = batch_input;  // Keep the real batch for the routing monitor.
            auto result = run_training_graph(batch_input, batch_target);
            step_loss_arr = multiply(result.loss, array((float)accum, float32));
#ifdef USE_CAPACITY_MOE
            step_aux_arr = multiply(result.aux, array((float)accum, float32));
#endif
            accum_grads = std::move(result.grads);
            std::vector<array> outputs = accum_grads;
            outputs.push_back(step_loss_arr);
#ifdef USE_CAPACITY_MOE
            outputs.push_back(step_aux_arr);
#endif
            eval(outputs);
        } else {
            for (int a = 0; a < accum; ++a) {
                auto [input, target] = sample_window();
                array micro_input = array(input.data(), {seq_len}, int32);
                array micro_target = array(target.data(), {seq_len}, int32);
                inp = micro_input;
                auto result = run_training_graph(micro_input, micro_target);
                step_loss_arr = add(step_loss_arr, result.loss);
#ifdef USE_CAPACITY_MOE
                step_aux_arr = add(step_aux_arr, result.aux);
#endif
                if (a == 0) accum_grads = std::move(result.grads);
                else for (size_t i = 0; i < result.grads.size(); ++i)
                    accum_grads[i] = add(accum_grads[i], result.grads[i]);

                if (a == accum - 1) {
                    std::vector<array> outputs = accum_grads;
                    outputs.push_back(step_loss_arr);
#ifdef USE_CAPACITY_MOE
                    outputs.push_back(step_aux_arr);
#endif
                    eval(outputs);
                }
            }
        }

        // 均一化梯度
        float grad_divisor = use_packed_batch ? 1.0f : static_cast<float>(accum);
        for (auto& g : accum_grads) g = divide(g, array(grad_divisor, float16));

        // 全局梯度大放水，卡在 5.0f 确保初始剧烈拟合
        array global_sq_norm = array(0.0f, float32);
        for (const auto& g : accum_grads) global_sq_norm = add(global_sq_norm, sum(square(astype(g, float32))));
        array gnorm = sqrt(global_sq_norm);
        array scale = where(greater(gnorm, array(5.0f, float32)), divide(array(5.0f, float32), gnorm), array(1.0f, float32));
        for (auto& g : accum_grads) g = multiply(g, astype(scale, float16));

        // 🔬 听诊器雷达：时刻盯着总梯度的活跃度

        // 调用优化器进行 FP32 绝对精度霸体更新
        // max_grad_norm 传 9999.0f（>= 9999 触发 _run_*_update 里的"跳过
        // 逐参数裁剪"分支）：全局裁剪已在上面以 5.0 完成，逐参数 norm 必然
        // ≤ 全局 norm ≤ 5.0 << 100，原来的 100.0f 实际是永不触发的 no-op，
        // 却在每个参数上白算一次 sqrt(sum(square(grad))) 全量归约。
        apply_snr_gated_update(model.mp, accum_grads, lr, b1_corr, b2_corr, 9999.0f, s, {gnorm});
        float total_g_norm = gnorm.item<float>();
        gnorm_history.push(total_g_norm);
        if (clear_interval > 0 && (s + 1) % clear_interval == 0) clear_cache();

        float avg_loss = step_loss_arr.item<float>() / accum;
        loss_history.push(avg_loss);
        loss_history_full.push_back(avg_loss);
        if (first_loss_recorded < 0.0f) first_loss_recorded = avg_loss;
#ifdef USE_CAPACITY_MOE
        float avg_aux = step_aux_arr.item<float>() / accum;
        aux_history.push(avg_aux);
        aux_history_full.push_back(avg_aux);
#endif

        if (s % 10 == 0) {
            // 📊 监控面板：清屏后重新打印完整仪表盘，而不是不断往下追加新的几行——
            // 这样屏幕上始终只有"最新一份"面板，不会随训练步数增多而持续刷屏滚动。
            // "\033[2J\033[H" 是标准ANSI转义码：\033[2J 清空整屏，\033[H 把光标移到左上角。
            //
            // 注意：自从加入负载均衡辅助损失后，grad_fn 反向传播实际优化的是
            // "总Loss"(任务loss + aux_loss_weight*aux_loss)。任务损失仍包含
            // cross_entropy_loss 的 confidence penalty；只读 eval 则报告纯交叉熵。
            // 分开展示，避免辅助损失的波动掩盖任务损失的变化。
            std::cout << "\033[2J\033[H";
            std::cout << "┌─ Step " << s << " ──────────────────────────────────\n";
#ifdef USE_CAPACITY_MOE
            float task_loss = avg_loss - aux_loss_weight * avg_aux;
            std::cout << "│ 总Loss(近60步) " << render_sparkline(loss_history.to_vector())
                      << "  " << std::fixed << std::setprecision(4) << avg_loss << "\n";
            std::cout << "│ 任务Loss(含置信度惩罚)  " << std::fixed << std::setprecision(4) << task_loss << "\n";
            std::cout << "│ Aux   " << render_sparkline(aux_history.to_vector())
                      << "  " << std::fixed << std::setprecision(4) << avg_aux
                      << " (权重=" << aux_loss_weight << ")\n";
            // 理论上完全均衡分布时 aux_loss = 1.0（n_experts * (1/n_experts)^2 * n_experts
            // 化简后正好是1.0，已用numpy验证过）。这里把这个参照值打出来，方便
            // 直接判断当前离"理想均衡"还有多远，而不只是看绝对数字没有概念。
            std::cout << "│ Aux(全程,共" << aux_history_full.size() << "步) "
                      << render_sparkline(aux_history_full)
                      << "  (理想均衡参照值=1.0)\n";
#else
            std::cout << "│ Loss(近60步) " << render_sparkline(loss_history.to_vector())
                      << "  " << std::fixed << std::setprecision(4) << avg_loss << "\n";
#endif
            // 全程趋势：不管训练跑了多少步，sparkline宽度始终恒定（downsample自动
            // 分桶压缩），用于一眼看出从训练开始到现在的整体走势，弥补上面"近60步"
            // 视图看不到长期趋势的局限。同时给出起点→当前的百分比变化作为参照数字。
            std::cout << "│ Loss(全程,共" << loss_history_full.size() << "步) "
                      << render_sparkline(loss_history_full) << "\n";
            if (first_loss_recorded > 1e-6f) {
                float pct_change = (avg_loss - first_loss_recorded) / first_loss_recorded * 100.0f;
                std::cout << "│   起点 " << std::fixed << std::setprecision(4) << first_loss_recorded
                          << "  →  当前 " << avg_loss
                          << "  (" << (pct_change <= 0 ? "↓" : "↑") << std::abs(pct_change) << "%)\n";
            }
            std::cout << "│ Grad  " << render_sparkline(gnorm_history.to_vector())
                      << "  " << std::fixed << std::setprecision(4) << total_g_norm << "\n";
            std::cout << "│ LR: " << std::scientific << lr << std::fixed << "\n";
            std::cout << "├────────────────────────────────────────────────\n";
            std::cout << last_expert_panel;
            std::cout << "└────────────────────────────────────────────────\n" << std::flush;
        }

#ifdef USE_CAPACITY_MOE
        // Low-frequency routing probe using the actual sampled batch. This
        // probes embeddings, not attention/loop hidden states, so its capacity
        // counts must not be presented as measured dispatch in the full model.
        //
        // 修复记录：初版这里直接把 inp（[seq_len] 的1D token id整数数组）传给
        // CapacityMoE::debug_fill_count，但该函数内部的 router 期望
        // [L, dim] 的浮点 embedding，导致 router(x) 内部 matmul/slice 形状
        // 不匹配，在 s=200 时崩溃报 "[slice] Invalid number of indices or
        // strides for array with dimension 1"。现改为调用 model 新增的
        // debug_moe_fill_count，内部先正确做 embedding lookup 再统计负载。
        if (s > 0 && s % expert_monitor_interval == 0) {
            array per_batch_fc = model.debug_moe_fill_count(inp);
            array fc = packed_batch_size > 1 ? sum(per_batch_fc, 0) : per_batch_fc;
            array peak_per_sequence = max(per_batch_fc);
            eval({fc, peak_per_sequence});
            std::vector<float> fill_counts(moe_experts);
            for (int i = 0; i < moe_experts; ++i) fill_counts[i] = fc.data<float>()[i];

            float max_load = *std::max_element(fill_counts.begin(), fill_counts.end());
            float max_sequence_load = peak_per_sequence.item<float>();
            float ideal_load = (float)(inp.shape(0) * moe_topk) / moe_experts;
            float imbalance_ratio = max_load / ideal_load;  // 1.0=完美均衡
            imbalance_ratio_history.push_back(imbalance_ratio);

            std::ostringstream panel;
            panel << "🧮 嵌入输入路由探针 (每样本capacity=" << model.recurrent.moe.capacity << ")\n";
            panel << render_horizontal_bars(fill_counts, 30);
            if (max_sequence_load > model.recurrent.moe.capacity) {
                panel << "    探针单样本最高负载(" << max_sequence_load
                      << ")超过capacity；完整隐层路由需另行统计\n";
            }
            // 不均衡趋势：每次采样只是一个瞬时快照，容易受随机噪声影响，
            // 看不出负载均衡损失是否真的在缓慢改善。这条线展示所有采样点
            // 的"最高负载/理论均衡负载"比值走势，如果整体趋势是下降的，
            // 说明 aux_loss 在起效，只是需要更多步数才能体现；如果一直
            // 持平不降，才是真正需要进一步调整权重或加路由噪声的信号。
            panel << "    不均衡趋势(共" << imbalance_ratio_history.size() << "次采样，1.0=完美均衡): "
                  << render_sparkline(imbalance_ratio_history)
                  << "  当前=" << std::fixed << std::setprecision(2) << imbalance_ratio << "\n";
            last_expert_panel = panel.str();
        }
#endif
        
        if (s > 0 && s % 100 == 0) {
            const int next_step = s + 1;
            mlx_checkpoint::save_atomic(ckpt_path, model_metadata, [&](std::ostream &os) {
                os.write(reinterpret_cast<const char*>(&next_step), sizeof(next_step));
                model.mp.save(os);
            });
            std::cout << "Checkpoint saved at completed step " << next_step << std::endl;
        }
    }
    if (get_int_arg("--save-final", 1) != 0) {
        const int completed = std::max(start_step, total_steps);
        mlx_checkpoint::save_atomic(ckpt_path, model_metadata, [&](std::ostream &os) {
            os.write(reinterpret_cast<const char*>(&completed), sizeof(completed));
            model.mp.save(os);
        });
        std::cout << "Final checkpoint saved at completed step " << completed << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}
