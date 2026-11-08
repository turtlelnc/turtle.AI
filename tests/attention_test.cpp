#define main turtle_training_main
#include "../train.cpp"
#undef main
#include <cassert>

void check_attention(int length, int window, int sinks) {
  AttentionLayer layer(length, 2, 2);
  layer.configure_context(window, sinks);
  Tensor input(length, 2), output(length, 2);
  for (int i = 0; i < length * 2; ++i) {
    input.data[i] = 0.1f * (i + 1);
    output.grad[i] = 0.2f + 0.03f * i;
  }
  for (auto linear : {&layer.W_q, &layer.W_k, &layer.W_v}) {
    for (int i = 0; i < 4; ++i) linear->W->data[i] = 0.1f * (i + 1);
    for (int i = 0; i < 2; ++i) linear->b->data[i] = 0;
    linear->clear_grad();
  }
  layer.forward(input, output);
  for (int q = 0; q < length; ++q) {
    float sum = 0;
    for (int k = 0; k < length; ++k) {
      float p = layer.scores_softmax.data[q * length + k];
      bool expected = k <= q && (window == 0 || k < sinks || q - k < window);
      assert(std::isfinite(p));
      if (!expected) assert(p == 0);
      else assert(p > 0);
      sum += p;
    }
    assert(std::abs(sum - 1) < 1e-6f);
  }
  layer.backward(input, output);
  std::vector<float> analytical(input.grad, input.grad + length * 2);
  // Finite differences catch missing Q/K/V gradient paths and mask mismatch.
  const float epsilon = 0.001f;
  auto objective = [&]() {
    layer.forward(input, output);
    float result = 0;
    for (int i = 0; i < length * 2; ++i) result += output.data[i] * output.grad[i];
    return result;
  };
  for (int i = 0; i < length * 2; ++i) {
    float original = input.data[i];
    input.data[i] = original + epsilon;
    float plus = objective();
    input.data[i] = original - epsilon;
    float minus = objective();
    input.data[i] = original;
    assert(std::abs(analytical[i] - (plus - minus) / (2 * epsilon)) < 0.002f);
  }
}

int main() {
  check_attention(6, 0, 8); // Default full causal attention.
  check_attention(6, 2, 1); // Old sink retained; middle history masked.
  check_attention(3, 1, 0); // Only self, and a different sequence length.
  check_attention(4, 20, 8); // Window/sinks may exceed sequence length.
  AttentionLayer layer(2, 2, 2);
  bool rejected = false;
  try { layer.configure_context(-1, 8); }
  catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
  std::cout << "Attention mask and finite-difference gradients passed\n";
}
