#define main turtle_training_main
#include "../train.cpp"
#undef main
#include <chrono>

namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

void close(float actual, float expected, float tolerance = 0.003f) {
  if (!std::isfinite(actual) || std::fabs(actual - expected) >= tolerance)
    std::cerr << "actual=" << actual << " expected=" << expected << '\n';
  require(std::isfinite(actual) && std::fabs(actual - expected) < tolerance,
          "Numerical result differs from reference");
}

template <class Forward>
void check_gradient(Tensor &input, Tensor &output, Forward forward, int stride = 1,
                    float epsilon = 0.0001f) {
  for (int i = 0; i < input.rows * input.cols; i += stride) {
    const float original = input.data[i];
    const float analytic = input.grad[i];
    auto objective = [&]() {
      forward();
      double result = 0;
      for (int j = 0; j < output.rows * output.cols; ++j)
        result += output.data[j] * output.grad[j];
      return result;
    };
    input.data[i] = original + epsilon;
    const double plus = objective();
    input.data[i] = original - epsilon;
    const double minus = objective();
    input.data[i] = original;
    close(analytic, static_cast<float>((plus - minus) / (2 * epsilon)));
  }
}

void test_attention(int length, int width) {
  AttentionLayer layer(length, width, width);
  Tensor input(length, width), output(length, width);
  for (int i = 0; i < length * width; ++i) {
    input.data[i] = 0.15f * (i + 1);
    output.grad[i] = 0.1f * (i % 3 + 1);
  }
  layer.clear_grad();
  layer.forward(input, output);
  layer.backward(input, output);
  check_gradient(input, output, [&]() { layer.forward(input, output); });
  // Also verify a parameter gradient, not just the shared input gradient.
  check_gradient(*layer.W_q.W, output, [&]() { layer.forward(input, output); });
}

void test_math() {
  // Exercise the OpenMP path with non-square matrices and uneven dimensions.
  Tensor large_x(33, 35), large_w(35, 37), large_y(33, 37);
  for (int i = 0; i < 33 * 35; ++i) large_x.data[i] = (i % 11 - 5) * 0.03f;
  for (int i = 0; i < 35 * 37; ++i) large_w.data[i] = (i % 7 - 3) * 0.04f;
  matmul(large_x.data, large_w.data, large_y.data, 33, 35, 37);
  for (int i = 0; i < 33; ++i)
    for (int j = 0; j < 37; ++j) {
      float expected = 0;
      for (int k = 0; k < 35; ++k)
        expected += large_x.data[i * 35 + k] * large_w.data[k * 37 + j];
      close(large_y.data[i * 37 + j], expected, 0.00001f);
    }
  Tensor x(2, 3), w(3, 4), y(2, 4), reference(2, 4);
  for (int i = 0; i < 6; ++i) x.data[i] = 0.2f * (i - 2);
  for (int i = 0; i < 12; ++i) w.data[i] = 0.1f * (i - 5);
  matmul(x.data, w.data, y.data, 2, 3, 4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j) {
      float expected = 0;
      for (int k = 0; k < 3; ++k)
        expected += x.data[i * 3 + k] * w.data[k * 4 + j];
      close(y.data[i * 4 + j], expected, 0.00001f);
      y.grad[i * 4 + j] = 0.1f * (j + 1);
    }
  linear_backward(x, w, y);
  check_gradient(x, y, [&]() { linear_forward(x, w, y); });
  check_gradient(w, y, [&]() { linear_forward(x, w, y); });
  mse_loss_backward(y, reference);
  for (int i = 0; i < 8; ++i) {
    float value = y.data[i];
    y.data[i] = value + 0.001f;
    float plus = mse_loss(y, reference);
    y.data[i] = value - 0.001f;
    float minus = mse_loss(y, reference);
    y.data[i] = value;
    close(y.grad[i], (plus - minus) / 0.002f);
  }
  // A second attention shape exposes buffers incorrectly shared between instances.
  std::cout << "Checking attention gradients\n";
  test_attention(2, 3);
  test_attention(4, 2);
  std::cout << "Checking transformer gradients\n";
  TransformerBlock block(2, 3, 3);
  Tensor input(2, 3), output(2, 3);
  const float values[] = {0.1f, -0.3f, 0.7f, 0.8f, -0.2f, 0.4f};
  std::copy(values, values + 6, input.data);
  for (int i = 0; i < 6; ++i) output.grad[i] = 0.2f * (i + 1);
  block.clear_grad();
  block.forward(input, output);
  block.backward(input, output);
  check_gradient(input, output, [&]() { block.forward(input, output); });

  block.clear_grad();
  for (auto *norm : {&block.norm1, &block.norm2}) {
    std::fill(norm->g_grad.begin(), norm->g_grad.end(), 0.5f);
    std::fill(norm->b_grad.begin(), norm->b_grad.end(), -0.25f);
  }
  block.update(0.1f);
  for (auto *norm : {&block.norm1, &block.norm2}) {
    for (float gamma : norm->gamma) close(gamma, 0.95f, 0.00001f);
    for (float beta : norm->beta) close(beta, 0.025f, 0.00001f);
  }

  LayerNorm norm(3, 2);
  Tensor normalized(2, 3);
  input.clear_grad();
  for (int i = 0; i < 6; ++i) normalized.grad[i] = 0.2f * (i + 1);
  norm.gamma = {0.8f, 1.1f, 1.2f};
  norm.forward(input, normalized);
  norm.backward(input, normalized);
  check_gradient(input, normalized, [&]() { norm.forward(input, normalized); });
  for (bool gamma : {true, false}) {
    Tensor parameters(1, 3);
    auto &values = gamma ? norm.gamma : norm.beta;
    const auto &gradient = gamma ? norm.g_grad : norm.b_grad;
    std::copy(values.begin(), values.end(), parameters.data);
    std::copy(gradient.begin(), gradient.end(), parameters.grad);
    check_gradient(parameters, normalized, [&]() {
      std::copy_n(parameters.data, 3, values.data());
      norm.forward(input, normalized);
    });
    std::copy_n(parameters.data, 3, values.data());
  }

  EmbeddingLayer embedding(260, 3);
  Tensor embeddings(2, 3);
  embedding.forward_ids({-1, 999}, embeddings);
  embedding.clear_grad();
  std::fill(embeddings.grad, embeddings.grad + 6, 1.0f);
  embedding.backward(embeddings, embeddings);
  for (int i = 0; i < 3; ++i) close(embedding.weights.grad[i], 2.0f);
}

void test_bpe(const fs::path &directory) {
  bpe::BPEConfig config;
  config.vocab_size = 280;
  bpe::BPETrainer trainer(config);
  trainer.train_from_texts({"banana banana banana", "hello hello"});
  const std::string sample = "hello banana 世界!\n";
  auto encoded = trainer.encode(sample);
  const auto fingerprint = trainer.fingerprint();
  require(trainer.decode(encoded) == sample, "BPE byte round trip failed");
  require(trainer.decode(trainer.encode(sample, true), true) == sample,
          "BPE special-token round trip failed");
  auto path = directory / "bpe.bin";
  require(trainer.save(path.string()), "BPE save failed");
  require(trainer.load(path.string()) && trainer.encode(sample) == encoded,
          "BPE reload failed");
  require(trainer.load(path.string()) && trainer.encode(sample) == encoded,
          "Repeated BPE load changed the model");
  require(trainer.fingerprint() == fingerprint, "BPE reload changed fingerprint");
  std::ifstream saved(path, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(saved)), {});
  auto broken = directory / "broken.bin";
  for (size_t cut : {size_t(0), size_t(1), sizeof(size_t), bytes.size() - 1}) {
    std::ofstream out(broken, std::ios::binary);
    out.write(bytes.data(), cut);
    out.close();
    require(!trainer.load(broken.string()), "Truncated BPE accepted");
    require(trainer.encode(sample) == encoded, "Failed load changed BPE state");
  }
  {
    std::ofstream out(broken, std::ios::binary);
    size_t impossible = std::numeric_limits<size_t>::max();
    out.write(reinterpret_cast<char *>(&impossible), sizeof(impossible));
  }
  require(!trainer.load(broken.string()), "Invalid BPE count accepted");
  {
    std::ofstream out(broken, std::ios::binary);
    bytes.back() ^= 0x7f; // Corrupt the final token ID while retaining valid lengths.
    out.write(bytes.data(), bytes.size());
  }
  require(!trainer.load(broken.string()), "Invalid BPE token ID accepted");
  trainer.train_from_texts({"zzzz zzzz zzzz"});
  bpe::BPETrainer fresh(config);
  fresh.train_from_texts({"zzzz zzzz zzzz"});
  require(trainer.fingerprint() == fresh.fingerprint() &&
          trainer.fingerprint() != fingerprint, "BPE fingerprint is inconsistent");
  require(trainer.encode(sample) == fresh.encode(sample) &&
          trainer.vocab_size() == fresh.vocab_size(), "Retraining kept old merges");
}

void test_image(const fs::path &directory) {
  auto path = directory / "pixel.ppm";
  {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n1 1\n255\n";
    const char pixel[] = {127, 64, 32};
    out.write(pixel, 3);
  }
  Tensor patches = image_to_patches(path.string(), 2, 1, 2);
  require(patches.rows == 4, "Single-pixel image conversion failed");
  for (int i = 0; i < 8; ++i)
    require(std::isfinite(patches.data[i]), "Image produced nonfinite values");

  Tensor raw(196, 768), image_embeddings(196, 3), input(198, 3);
  LinearLayer projection(768, 3);
  process_image_to_input(path.string(), projection, input, raw, image_embeddings);
  for (int i = 0; i < 196 * 768; ++i)
    close(raw.data[i], static_cast<unsigned char>("\177\100\040"[i % 3]) / 255.0f, 0.00001f);
  projection.clear_grad();
  std::fill(image_embeddings.grad, image_embeddings.grad + 196 * 3, 0.01f);
  projection.backward(raw, image_embeddings);
  check_gradient(*projection.W, image_embeddings,
                 [&]() { projection.forward(raw, image_embeddings); }, 383, 0.001f);
  for (int i = 0; i < 3; ++i) close(projection.b->grad[i], 1.96f, 0.0001f);

  // Distinct pixels must remain distinct patches in spatial order.
  auto grid = directory / "grid.ppm";
  {
    std::ofstream out(grid, std::ios::binary);
    out << "P6\n2 2\n255\n";
    const unsigned char pixels[] = {255,0,0, 0,255,0, 0,0,255, 255,255,255};
    out.write(reinterpret_cast<const char *>(pixels), sizeof(pixels));
  }
  Tensor spatial(4, 3);
  extract_image_patches(grid.string(), spatial, 1, 2);
  const float expected[] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
  for (int i = 0; i < 12; ++i) close(spatial.data[i], expected[i], 0.00001f);
}

void test_checkpoint(const fs::path &directory) {
  bpe::BPETrainer tokenizer;
  EmbeddingLayer embed(260, 3);
  LinearLayer vision(768, 3), projection(3, 260);
  std::vector<std::unique_ptr<TransformerBlock>> blocks;
  blocks.push_back(std::make_unique<TransformerBlock>(2, 3, 3));
  for (auto *norm : {&blocks[0]->norm1, &blocks[0]->norm2}) {
    norm->gamma = {0.8f, 1.1f, 1.2f};
    norm->beta = {0.2f, -0.1f, 0.4f};
  }
  auto state = model_state(embed, vision, blocks, projection, 2, tokenizer);
  auto predictions = [&]() {
    Tensor input(2, 3), hidden(2, 3), logits(2, 260);
    embed.forward_ids({4, 5}, input);
    blocks[0]->forward(input, hidden);
    projection.forward(hidden, logits);
    return std::vector<float>(logits.data, logits.data + 520);
  };
  const auto expected_predictions = predictions();
  std::vector<std::vector<float>> expected;
  for (const auto &[data, count] : state.parameters) expected.emplace_back(data, data + count);
  auto path = (directory / "model.bin").string();
  checkpoint::save(path, state);
  for (const auto &[data, count] : state.parameters) std::fill_n(data, count, 0.0f);
  require(!checkpoint::load(path, state), "New checkpoint was interpreted as legacy");
  for (size_t i = 0; i < state.parameters.size(); ++i)
    require(std::equal(expected[i].begin(), expected[i].end(), state.parameters[i].first),
            "Checkpoint did not restore all parameters");
  require(predictions() == expected_predictions, "Reloaded model predictions changed");

  auto mismatched = state;
  mismatched.metadata["seq_len"] = 3;
  bool rejected = false;
  try { checkpoint::load(path, mismatched); }
  catch (const std::exception &) { rejected = true; }
  require(rejected && predictions() == expected_predictions,
          "Metadata mismatch mutated the model");

  std::ifstream saved(path, std::ios::binary);
  saved.seekg(8);
  const size_t payload_offset = 12 + checkpoint::read_length(saved);
  saved.seekg(0);
  std::string bytes((std::istreambuf_iterator<char>(saved)), {});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(bytes.data() + payload_offset, &nan, sizeof(nan));
  auto invalid = (directory / "nonfinite.bin").string();
  {
    std::ofstream out(invalid, std::ios::binary);
    out.write(bytes.data(), bytes.size());
  }
  rejected = false;
  try { checkpoint::load(invalid, state); }
  catch (const std::exception &) { rejected = true; }
  require(rejected && predictions() == expected_predictions,
          "Nonfinite checkpoint mutated the model");

  // Failed saves preserve the last complete checkpoint.
  state.parameters.back().first[0] = std::numeric_limits<float>::quiet_NaN();
  rejected = false;
  try { checkpoint::save(path, state); }
  catch (const std::exception &) { rejected = true; }
  require(rejected, "Nonfinite parameters were saved");
  checkpoint::load(path, state);
  require(predictions() == expected_predictions, "Failed save destroyed checkpoint");

  auto legacy = (directory / "legacy.bin").string();
  {
    std::ofstream out(legacy, std::ios::binary);
    for (size_t i = 0; i < state.legacy_blocks; ++i)
      out.write(reinterpret_cast<const char *>(expected[i].data()), expected[i].size() * sizeof(float));
  }
  require(checkpoint::load(legacy, state), "Legacy checkpoint failed to load");
  require(predictions() == expected_predictions, "Legacy parameters loaded incorrectly");
}
} // namespace

int main() {
  auto directory = fs::temp_directory_path() /
      ("turtle-regression-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  try {
    srand(1);
    test_math();
    test_bpe(directory);
    test_image(directory);
    test_checkpoint(directory);
    fs::remove_all(directory);
    std::cout << "Regression tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    fs::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
