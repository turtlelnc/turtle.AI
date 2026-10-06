#ifndef TURTLE_MLX_CHECKPOINT_IO_H
#define TURTLE_MLX_CHECKPOINT_IO_H
#include "../checkpoint.h"
#include <functional>
#include <iostream>

namespace mlx_checkpoint {
inline constexpr char magic[] = "TRTLMX01";
inline void write_metadata(std::ostream &out, const nlohmann::json &metadata) {
    const std::string text = metadata.dump();
    if (text.size() > checkpoint::max_header_size) throw std::runtime_error("MLX checkpoint metadata is too large");
    out.write(magic, 8);
    checkpoint::write_length(out, static_cast<uint32_t>(text.size()));
    out.write(text.data(), text.size());
}
inline bool read_metadata(std::istream &in, const nlohmann::json &expected) {
    const auto start = in.tellg();
    char signature[8];
    if (!in.read(signature, 8)) throw std::runtime_error("MLX checkpoint is truncated");
    if (!std::equal(signature, signature + 8, magic)) {
        in.seekg(start);
        std::cerr << "Warning: legacy MLX checkpoint has no architecture/tokenizer metadata.\n";
        return false;
    }
    const auto length = checkpoint::read_length(in);
    if (length == 0 || length > checkpoint::max_header_size) throw std::runtime_error("Invalid MLX checkpoint header length");
    std::string text(length, '\0');
    if (!in.read(text.data(), length)) throw std::runtime_error("MLX checkpoint metadata is truncated");
    const auto actual = nlohmann::json::parse(text);
    for (auto it = expected.begin(); it != expected.end(); ++it)
        if (!actual.contains(it.key()) || actual.at(it.key()) != it.value())
            throw std::runtime_error("MLX checkpoint metadata mismatch: " + it.key());
    return true;
}
inline void save_atomic(const std::string &path, const nlohmann::json &metadata,
                        const std::function<void(std::ostream &)> &writer) {
    const auto temporary = path + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    try {
        std::ofstream out(temporary, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write MLX checkpoint: " + path);
        write_metadata(out, metadata);
        writer(out);
        out.close();
        if (!out) throw std::runtime_error("Failed writing MLX checkpoint: " + path);
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
}
#endif
