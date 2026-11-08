#ifndef TURTLE_MEMORY_H
#define TURTLE_MEMORY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace turtle {
namespace memory {

// Storage policy only. Arithmetic and returned vectors are always FP32.
enum class Precision { FP32, INT8, INT4 };

class StoredVector {
  Precision precision_;
  std::size_t size_;
  float scale_ = 1.0f;
  std::vector<float> full_;
  std::vector<std::uint8_t> packed_;

public:
  StoredVector(const std::vector<float>& values, Precision precision)
      : precision_(precision), size_(values.size()) {
    if (values.empty()) throw std::invalid_argument("empty memory vector");
    float maximum = 0;
    for (float value : values) {
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite memory vector");
      maximum = std::max(maximum, std::abs(value));
    }
    if (precision == Precision::FP32) {
      full_ = values;
      return;
    }
    if (precision != Precision::INT8 && precision != Precision::INT4)
      throw std::invalid_argument("unsupported precision");
    const int limit = precision == Precision::INT8 ? 127 : 7;
    // A positive scale also handles zero and subnormal vectors safely.
    scale_ = std::max(maximum / limit, std::numeric_limits<float>::min());
    packed_.resize(precision == Precision::INT8 ? size_ : (size_ + 1) / 2, 0);
    for (std::size_t i = 0; i < size_; ++i) {
      int q = static_cast<int>(std::round(values[i] / scale_));
      q = std::max(-limit, std::min(limit, q));
      if (precision == Precision::INT8)
        packed_[i] = static_cast<std::uint8_t>(q + 127);
      else
        packed_[i / 2] |= static_cast<std::uint8_t>((q + 7) << (4 * (i % 2)));
    }
  }

  std::size_t size() const { return size_; }
  Precision precision() const { return precision_; }
  float scale() const { return scale_; }
  // Encoded payload plus quantization scale, excluding container metadata.
  std::size_t storage_bytes() const {
    return precision_ == Precision::FP32 ? full_.size() * sizeof(float)
                                        : packed_.size() + sizeof(scale_);
  }
  std::vector<float> decode() const {
    if (precision_ == Precision::FP32) return full_;
    std::vector<float> result(size_);
    for (std::size_t i = 0; i < size_; ++i) {
      int q = precision_ == Precision::INT8
                  ? static_cast<int>(packed_[i]) - 127
                  : static_cast<int>((packed_[i / 2] >> (4 * (i % 2))) & 15) - 7;
      result[i] = q * scale_;
    }
    return result;
  }
};

// Half-open source positions. Callers choose boundaries and supply representations.
struct Block {
  std::uint64_t id;
  std::size_t begin;
  std::size_t end;
  StoredVector key;
  StoredVector value;
};

struct Hit {
  std::uint64_t id;
  std::size_t begin;
  std::size_t end;
  float score;
  std::vector<float> value;
};

class Store {
  std::size_t key_dim_, value_dim_, capacity_;
  std::vector<Block> blocks_;

public:
  Store(std::size_t key_dim, std::size_t value_dim, std::size_t capacity)
      : key_dim_(key_dim), value_dim_(value_dim), capacity_(capacity) {
    if (!key_dim || !value_dim || !capacity)
      throw std::invalid_argument("memory dimensions/capacity must be positive");
  }
  std::size_t size() const { return blocks_.size(); }
  std::size_t storage_bytes() const {
    std::size_t bytes = 0;
    for (const auto& block : blocks_)
      bytes += block.key.storage_bytes() + block.value.storage_bytes();
    return bytes;
  }
  void clear() { blocks_.clear(); }
  // Explicit admission: never silently evict a protected or important block.
  void add(Block block) {
    if (block.begin >= block.end || block.key.size() != key_dim_ ||
        block.value.size() != value_dim_)
      throw std::invalid_argument("invalid memory block boundaries/dimensions");
    for (const auto& existing : blocks_)
      if (existing.id == block.id) throw std::invalid_argument("duplicate memory id");
    if (blocks_.size() == capacity_) throw std::length_error("memory capacity reached");
    blocks_.push_back(std::move(block));
  }
  // Visible blocks must end before or at the query position. No future leakage.
  // Exact MIPS baseline, not an approximate index or a learned RAG pipeline.
  std::vector<Hit> search(const std::vector<float>& query, std::size_t query_position,
                          std::size_t top_k) const {
    if (query.size() != key_dim_) throw std::invalid_argument("query dimension mismatch");
    for (float value : query)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite query");
    std::vector<std::pair<float, std::size_t>> candidates;
    if (!top_k) return {};
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      if (blocks_[i].end > query_position) continue;
      auto key = blocks_[i].key.decode();
      float score = 0;
      for (std::size_t j = 0; j < key_dim_; ++j) score += query[j] * key[j];
      if (!std::isfinite(score)) throw std::overflow_error("memory score overflow");
      candidates.emplace_back(score, i);
    }
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first > b.first;
      return blocks_[a.second].id < blocks_[b.second].id;
    });
    std::vector<Hit> hits;
    for (std::size_t i = 0; i < std::min(top_k, candidates.size()); ++i) {
      const auto& block = blocks_[candidates[i].second];
      hits.push_back({block.id, block.begin, block.end, candidates[i].first,
                      block.value.decode()});
    }
    return hits;
  }
};

} // namespace memory
} // namespace turtle
#endif
