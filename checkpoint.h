#ifndef TURTLE_CHECKPOINT_H
#define TURTLE_CHECKPOINT_H

#include "json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace checkpoint {
inline constexpr char magic[] = "TRTLMODL";
inline constexpr size_t max_header_size = 65536;

// Parameter blocks keep the original payload order; new LayerNorm blocks follow it.
struct State {
  nlohmann::json metadata;
  std::vector<std::pair<float *, size_t>> parameters;
  size_t legacy_blocks = 0;

  size_t count(size_t blocks) const {
    size_t result = 0;
    for (size_t i = 0; i < blocks; ++i) {
      if (parameters[i].second > std::numeric_limits<size_t>::max() - result)
        throw std::runtime_error("Checkpoint parameter count overflows");
      result += parameters[i].second;
    }
    return result;
  }
};

inline std::string byte_order() {
  const uint16_t value = 1;
  return *reinterpret_cast<const unsigned char *>(&value) == 1 ? "little" : "big";
}

inline void write_length(std::ostream &out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}

inline uint32_t read_length(std::istream &in) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof())
      throw std::runtime_error("Checkpoint is truncated in the header");
    value |= static_cast<uint32_t>(byte) << (8 * i);
  }
  return value;
}

inline void save(const std::string &path, const State &state) {
  const std::string header = state.metadata.dump();
  if (header.size() > max_header_size)
    throw std::runtime_error("Checkpoint header is too large");
  for (const auto &[data, count] : state.parameters)
    for (size_t i = 0; i < count; ++i)
      if (!std::isfinite(data[i]))
        throw std::runtime_error("Cannot save checkpoint containing NaN/Inf");

  static std::atomic<uint64_t> serial{0};
  const std::string temporary = path + ".tmp-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
      std::to_string(serial++);
  try {
    std::ofstream out(temporary, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open checkpoint for writing: " + path);
    out.write(magic, 8);
    write_length(out, static_cast<uint32_t>(header.size()));
    out.write(header.data(), header.size());
    for (const auto &[data, count] : state.parameters)
      out.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    out.close();
    if (!out) throw std::runtime_error("Failed to write checkpoint: " + path);
    // Commit only a complete file, preserving the last checkpoint on failure.
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

// Returns true for legacy files. Metadata and all floats are checked before mutation.
inline bool load(const std::string &path, const State &state) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("Cannot open requested checkpoint: " + path);
  const auto end = in.tellg();
  if (end < 8) throw std::runtime_error("Checkpoint is truncated: " + path);
  in.seekg(0);
  char signature[8];
  in.read(signature, 8);
  const bool legacy = !std::equal(signature, signature + 8, magic);
  size_t blocks = state.parameters.size();
  if (legacy) {
    in.seekg(0);
    blocks = state.legacy_blocks;
  } else {
    const uint32_t length = read_length(in);
    if (length == 0 || length > max_header_size ||
        static_cast<std::streamoff>(length) > end - in.tellg())
      throw std::runtime_error("Checkpoint header length is invalid");
    std::string header(length, '\0');
    in.read(header.data(), length);
    const auto metadata = nlohmann::json::parse(header);
    if (!metadata.is_object()) throw std::runtime_error("Checkpoint metadata is invalid");
    for (auto it = state.metadata.begin(); it != state.metadata.end(); ++it)
      if (!metadata.contains(it.key()) || metadata.at(it.key()) != it.value())
        throw std::runtime_error("Checkpoint metadata mismatch: " + it.key());
  }
  const size_t count = state.count(blocks);
  if (count > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / sizeof(float) ||
      end - in.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
    throw std::runtime_error("Checkpoint is truncated or incompatible: " + path);
  std::vector<float> staged(count);
  if (!in.read(reinterpret_cast<char *>(staged.data()), count * sizeof(float)))
    throw std::runtime_error("Checkpoint read failed: " + path);
  for (float value : staged)
    if (!std::isfinite(value))
      throw std::runtime_error("Checkpoint contains NaN/Inf: " + path);
  size_t offset = 0;
  for (size_t i = 0; i < blocks; ++i) {
    const auto [data, size] = state.parameters[i];
    std::copy_n(staged.data() + offset, size, data);
    offset += size;
  }
  return legacy;
}
} // namespace checkpoint
#endif
