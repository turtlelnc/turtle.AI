#include "../memory.h"
#include <cassert>
#include <iostream>
using namespace turtle::memory;

template<class Error, class Fn> void rejects(Fn fn) {
  bool rejected = false;
  try { fn(); } catch (const Error&) { rejected = true; }
  assert(rejected);
}
int main() {
  std::vector<float> values{-2, -1, 0, 1, 2}; // Odd INT4 payload length.
  for (auto precision : {Precision::FP32, Precision::INT8, Precision::INT4}) {
    StoredVector encoded(values, precision);
    auto decoded = encoded.decode();
    for (std::size_t i = 0; i < values.size(); ++i)
      assert(std::abs(decoded[i] - values[i]) <= encoded.scale() / 2 + 1e-6f);
    if (precision == Precision::FP32) assert(decoded == values);
    StoredVector zeros(std::vector<float>(5, 0), precision);
    assert(zeros.decode() == std::vector<float>(5, 0));
  }
  assert(StoredVector(values, Precision::INT4).storage_bytes() == 7);
  rejects<std::invalid_argument>([] { StoredVector v({}, Precision::INT4); });
  rejects<std::invalid_argument>([] {
    StoredVector v({std::numeric_limits<float>::infinity()}, Precision::INT8);
  });
  Store store(2, 3, 3);
  store.add({2, 0, 4, StoredVector({1, 0}, Precision::FP32),
                         StoredVector({-2, 0, 2}, Precision::INT4)});
  store.add({1, 4, 8, StoredVector({1, 0}, Precision::INT8),
                         StoredVector({1, 2, 3}, Precision::FP32)});
  store.add({3, 8, 12, StoredVector({10, 0}, Precision::FP32),
                          StoredVector({4, 5, 6}, Precision::INT8)});
  assert(store.search({1, 0}, 3, 2).empty());
  auto hits = store.search({1, 0}, 8, 10);
  assert(hits.size() == 2 && hits[0].id == 1 && hits[1].id == 2);
  assert(hits[1].value == std::vector<float>({-2, 0, 2}));
  assert(store.search({1, 0}, 12, 1)[0].id == 3);
  assert(store.search({1, 0}, 12, 0).empty());
  rejects<std::invalid_argument>([&] { store.search({1}, 12, 1); });
  rejects<std::invalid_argument>([&] {
    store.add({2, 0, 4, StoredVector({1, 0}, Precision::FP32),
                         StoredVector({1, 2, 3}, Precision::FP32)});
  });
  rejects<std::length_error>([&] {
    store.add({4, 12, 16, StoredVector({1, 0}, Precision::FP32),
                           StoredVector({1, 2, 3}, Precision::FP32)});
  });
  store.clear();
  assert(store.size() == 0 && store.storage_bytes() == 0);
  std::cout << "Memory quantization, retrieval and causal isolation passed\n";
}
