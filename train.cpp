#include "BPE.h"
#include "json.hpp"
#include "checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#else
#define USE_ACCELERATE 0
#endif
using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;
struct Tensor {
  float *grad;
  float *data;
  int rows;
  int cols;
  Tensor(int r, int c) : rows(r), cols(c) {
    if (r < 0 || c < 0 || static_cast<long long>(r) * c > std::numeric_limits<int>::max())
      throw std::invalid_argument("Tensor dimensions exceed supported range");
    const size_t count = static_cast<size_t>(r) * c;
    auto values = std::make_unique<float[]>(count);
    auto gradients = std::make_unique<float[]>(count);
    data = values.release();
    grad = gradients.release();
  }
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&other) noexcept
      : grad(other.grad), data(other.data), rows(other.rows), cols(other.cols) {
    other.data = nullptr;
    other.grad = nullptr;
  }
  ~Tensor() {
    delete[] data;
    delete[] grad;
  }
  void save(std::ofstream &out) {
    out.write((char *)data, rows * cols * sizeof(float));
  }
  void load(std::ifstream &in) {
    in.read((char *)data, rows * cols * sizeof(float));
  }
  void clear_grad() {
    if (grad) {
        // 使用 memset 是最快且最彻底的
        std::memset(grad, 0, rows * cols * sizeof(float));
    }
}
  void update(float lr) {
    for (int i = 0; i < rows * cols; i++) {
      float g = grad[i];
      if (g > 1.0f)
        g = 1.0f; // 强制裁剪
      if (g < -1.0f)
        g = -1.0f;
      data[i] -= lr * g;
    }
  }
};
void transpose(float *a, float *b, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      b[j * m + i] = a[i * n + j];
    }
  }
}
void softmax(float *x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++)
    if (x[i] > m)
      m = x[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    x[i] = exp(x[i] - m);
    sum += x[i];
  }
  for (int i = 0; i < n; i++)
    x[i] /= sum;
}
void matmul(float *a, float *b, float *c, int m, int k, int n) {
#if USE_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b,
              n, 0.0f, c, n);
#else
#pragma omp parallel for if(static_cast<long long>(m) * k * n >= 32768)
  for (int i = 0; i < m; i++) {
    float *row = c + i * n;
    std::fill(row, row + n, 0.0f);
    for (int f = 0; f < k; f++) {
      const float value = a[i * k + f];
      const float *b_row = b + f * n;
#pragma omp simd
      for (int j = 0; j < n; j++)
        row[j] += value * b_row[j];
    }
  }
#endif
}
void linear_forward(Tensor &X, Tensor &W, Tensor &Y) {
  matmul(X.data, W.data, Y.data, X.rows, X.cols, W.cols);
}
void linear_backward(Tensor &X, Tensor &W, Tensor &Y) {
  Tensor x_tp(X.cols, X.rows);
  transpose(X.data, x_tp.data, X.rows, X.cols);
  matmul(x_tp.data, Y.grad, W.grad, X.cols, X.rows, Y.cols);
  Tensor w_tp(W.cols, W.rows);
  transpose(W.data, w_tp.data, W.rows, W.cols);
  matmul(Y.grad, w_tp.data, X.grad, Y.rows, Y.cols, W.rows);
}
float mse_loss(Tensor &Y_pred, Tensor &Y_target) {
  float tot_loss = 0.0f;
  int size = Y_pred.rows * Y_pred.cols;
  for (int i = 0; i < size; i++) {
    float diff = Y_pred.data[i] - Y_target.data[i];
    tot_loss += 0.5 * diff * diff;
  }
  return tot_loss / size;
}
void mse_loss_backward(Tensor &Y_pred, Tensor &Y_target) {
  for (int i = 0; i < Y_pred.rows * Y_pred.cols; i++) {
    Y_pred.grad[i] = (Y_pred.data[i] - Y_target.data[i]) /
                     (Y_pred.rows * Y_pred.cols);
  }
}
void relu_forward(Tensor &X, Tensor &Y) {
  for (int i = 0; i < X.rows * X.cols; i++) {
    Y.data[i] = max(X.data[i], 0.0f);
  }
}
void relu_backward(Tensor &X, Tensor &Y) {
  for (int i = 0; i < X.rows * X.cols; i++) {
    X.grad[i] = (X.data[i] > 0) ? Y.grad[i] : 0;
  }
}
void embedding_forward(int *input_ids, Tensor &weight, Tensor &output,
                       int seq_len) {
  int d_model = weight.cols;
  for (int i = 0; i < seq_len; i++) {
    int id = input_ids[i];
    memcpy(&output.data[i * d_model], &weight.data[id * d_model],
           sizeof(float) * d_model);
  }
}
void embedding_backward(int *input_ids, Tensor &weight, Tensor &grad_output,
                        int seq_len) {
  int d_model = weight.cols;
  for (int i = 0; i < seq_len; i++) {
    int id = input_ids[i];
    for (int j = 0; j < d_model; j++) {
      weight.grad[id * d_model + j] += grad_output.data[i * d_model + j];
    }
  }
}
class Layer {
public:
  virtual ~Layer() {}
  virtual void forward(Tensor &input, Tensor &output) = 0;
  virtual void backward(Tensor &input, Tensor &output) = 0;
  virtual void update(float lr) {}
  virtual void clear_grad() {}
};
class LinearLayer : public Layer {
public:
  Tensor *W, *b;
  int in_dim, out_dim;
  std::vector<float> x_transpose, w_transpose, input_gradient;

  LinearLayer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim) {
    x_transpose.reserve(in_dim);
    // 正确的堆内存分配方式
    W = new Tensor(in_dim, out_dim);
    b = new Tensor(1, out_dim);
    w_transpose.resize(in_dim * out_dim);

    float scale = sqrt(2.0f / in_dim);
    for (int i = 0; i < W->rows * W->cols; i++) {
      W->data[i] = ((rand() / (float)RAND_MAX) - 0.5f) * scale;
    }
    for (int i = 0; i < b->cols; i++)
      b->data[i] = 0.0f;
  }

  // 必须加上析构函数，否则会内存泄漏
  ~LinearLayer() {
    delete W;
    delete b;
  }

  void forward(Tensor &input, Tensor &output) override {
    // 注意：W 现在是指针，要用 W->data
    matmul(input.data, W->data, output.data, input.rows, input.cols, W->cols);

#if !USE_ACCELERATE
#pragma omp parallel for if(output.rows * output.cols >= 32768)
#endif
    for (int i = 0; i < output.rows; i++) {
      for (int j = 0; j < output.cols; j++) {
        output.data[i * output.cols + j] += b->data[j];
      }
    }
  }

  void backward(Tensor &input, Tensor &output) override {
    int M = input.rows;
    int K = input.cols;
    int N = output.cols;

    // Reuse buffers across steps and keep each projection's input gradient.
    x_transpose.resize(M * K);
    input_gradient.resize(M * K);
    transpose(input.data, x_transpose.data(), M, K);
    matmul(x_transpose.data(), output.grad, W->grad, K, M, N);
    transpose(W->data, w_transpose.data(), K, N);
    matmul(output.grad, w_transpose.data(), input_gradient.data(), M, N, K);
    std::copy(input_gradient.begin(), input_gradient.end(), input.grad);

    // Bias 梯度更新
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        b->grad[j] += output.grad[i * N + j];
      }
    }


  }

  void save(std::ofstream &out) {
    W->save(out);
    b->save(out); // 别忘了存 bias
  }

  void load(std::ifstream &in) {
    W->load(in);
    b->load(in); // 别忘了读 bias
  }

  void update(float lr) override {
    W->update(lr);
    for (int i = 0; i < b->cols; i++)
      b->data[i] -= lr * b->grad[i];
  }

  void clear_grad() override {
    W->clear_grad();
    for (int i = 0; i < b->cols; i++)
      b->grad[i] = 0.0f;
  }
};
class Sequential {
public:
  vector<Layer *> layers;
  vector<Tensor *> intermediates;
  void add(Layer *l, int out_rows, int out_cols) {
    layers.push_back(l);
    intermediates.push_back(new Tensor(out_rows, out_cols));
  }
  void forward(Tensor &input) {
    Tensor *current_input = &input;
    for (int i = 0; i < layers.size(); i++) {
      layers[i]->forward(*current_input, *intermediates[i]);
      current_input = intermediates[i];
    }
  }
  void backward(Tensor &input) {
    Tensor *current_output_grad = intermediates.back();
    for (int i = layers.size() - 1; i >= 0; i--) {
      Tensor *current_input = (i == 0) ? &input : intermediates[i - 1];
      layers[i]->backward(*current_input, *intermediates[i]);
    }
  }
  void update(float lr) {
    for (auto l : layers)
      l->update(lr);
  }
  void clear_grad() {
    for (auto l : layers)
      l->clear_grad();
    for (auto t : intermediates)
      t->clear_grad();
  }
  Tensor &get_output() { return *intermediates.back(); }
};
class ReLULayer : public Layer {
public:
  void forward(Tensor &input, Tensor &output) override {
    for (int i = 0; i < input.rows * input.cols; i++) {
      output.data[i] = std::max(input.data[i], 0.0f);
    }
  }
  void backward(Tensor &input, Tensor &output) override {
    for (int i = 0; i < input.rows * input.cols; i++) {
      input.grad[i] = (input.data[i] > 0) ? output.grad[i] : 0.0f;
    }
  }

  void update(float lr) override {}
  void clear_grad() override {}
};
class AttentionLayer : public Layer {
public:
  LinearLayer W_q, W_k, W_v;
  int seq_len;
  int d_model;
  int d_head;
  Tensor Q, K, V;
  Tensor K_tp, V_tp, scores_softmax_tp;
  Tensor scores, scores_softmax;
  Tensor d_scores_softmax, d_scores_raw, d_scores_raw_tp;

  AttentionLayer(int s, int m, int h)
      : W_q(m, h), W_k(m, h), W_v(m, h), seq_len(s), d_model(m), d_head(h),
        Q(s, h), K(s, h), V(s, h), K_tp(h, s), V_tp(h, s), scores_softmax_tp(s, s), scores(s, s),
        scores_softmax(s, s), d_scores_softmax(s, s), d_scores_raw(s, s),
        d_scores_raw_tp(s, s) {}

  void forward(Tensor &input, Tensor &output) override {
    W_q.forward(input, Q);
    W_k.forward(input, K);
    W_v.forward(input, V);
    transpose(K.data, K_tp.data, K.rows, K.cols);
    matmul(Q.data, K_tp.data, scores.data, seq_len, d_head, seq_len);

    float scale = 1.0f / sqrt((float)d_head);
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        if (j > i) {
          scores.data[i * seq_len + j] = -1e9f;
        } else {
          scores.data[i * seq_len + j] *= scale;
        }
      }
      softmax(&scores.data[i * seq_len], seq_len);
      memcpy(&scores_softmax.data[i * seq_len], &scores.data[i * seq_len],
             sizeof(float) * seq_len);
    }
    matmul(scores_softmax.data, V.data, output.data, seq_len, seq_len, d_head);
  }

  void backward(Tensor &input, Tensor &output) override {
    int L = seq_len;
    int D = d_head;
    transpose(scores_softmax.data, scores_softmax_tp.data, L, L);
    matmul(scores_softmax_tp.data, output.grad, V.grad, L, L, D);
    transpose(V.data, V_tp.data, L, D);
    matmul(output.grad, V_tp.data, d_scores_softmax.data, L, D, L);
    float scale = 1.0f / sqrt((float)D);
    for (int i = 0; i < L; i++) {
      float dot_sum = 0;
      for (int j = 0; j < L; j++) {
        dot_sum +=
            scores_softmax.data[i * L + j] * d_scores_softmax.data[i * L + j];
      }
      for (int j = 0; j < L; j++) {
        float y = scores_softmax.data[i * L + j];
        float dy = d_scores_softmax.data[i * L + j];
        d_scores_raw.data[i * L + j] = y * (dy - dot_sum) * scale;
        if (j > i)
          d_scores_raw.data[i * L + j] = 0.0f;
      }
    }

    matmul(d_scores_raw.data, K.data, Q.grad, L, L, D);
    transpose(d_scores_raw.data, d_scores_raw_tp.data, L, L);
    matmul(d_scores_raw_tp.data, Q.data, K.grad, L, L, D);
    W_q.backward(input, Q);
    W_k.backward(input, K);
    W_v.backward(input, V);
    // Q, K and V all depend on the same input; sum their contributions.
    for (int i = 0; i < input.rows * input.cols; ++i) {
      input.grad[i] = W_q.input_gradient[i] + W_k.input_gradient[i] +
                      W_v.input_gradient[i];
    }
  }

  void update(float lr) override {
    W_q.update(lr);
    W_k.update(lr);
    W_v.update(lr);
  }

  void clear_grad() override {
    W_q.clear_grad();
    W_k.clear_grad();
    W_v.clear_grad();
    Q.clear_grad();
    K.clear_grad();
    V.clear_grad();
  }

  void save(std::ofstream &out) {
    W_q.save(out);
    W_k.save(out);
    W_v.save(out);
  }

  void load(std::ifstream &in) {
    W_q.load(in);
    W_k.load(in);
    W_v.load(in);
  }
};
class EmbeddingLayer : public Layer {
public:
  Tensor weights;
  int vocab_size;
  int d_model;
  std::vector<int> last_input_ids;

  EmbeddingLayer(int v_size, int d_mod)
      : vocab_size(v_size), d_model(d_mod), weights(v_size, d_mod) {
    float scale = sqrt(2.0f / d_mod);
    for (int i = 0; i < v_size * d_mod; i++) {
      weights.data[i] = ((rand() / (float)RAND_MAX) - 0.5f) * scale;
    }
  }
  void forward(Tensor &input, Tensor &output) override {
    std::vector<int> ids;
    for (int i = 0; i < input.rows; i++) {
      ids.push_back((int)input.data[i * input.cols]);
    }
    forward_ids(ids, output);
  }
  void forward_ids(const std::vector<int> &token_ids, Tensor &output) {
    if (token_ids.size() > static_cast<size_t>(output.rows) || output.cols != d_model)
      throw std::invalid_argument("Embedding output shape does not match token IDs");
    last_input_ids = token_ids;
    std::fill(output.data, output.data + output.rows * output.cols, 0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++) {
      int id = token_ids[i];
      if (id < 0 || id >= vocab_size)
        id = 0;
      last_input_ids[i] = id;
      memcpy(&output.data[i * d_model], &weights.data[id * d_model],
             sizeof(float) * d_model);
    }
  }
  void backward(Tensor &input, Tensor &output) override {
    for (int i = 0; i < (int)last_input_ids.size(); i++) {
      int id = last_input_ids[i];
      for (int d = 0; d < d_model; d++) {
        weights.grad[id * d_model + d] += output.grad[i * d_model + d];
      }
    }
  }

  void update(float lr) override { weights.update(lr); }
  void clear_grad() override { weights.clear_grad(); }
};
class LayerNorm : public Layer {
public:
  int d_model;
  int max_seq_len;
  std::vector<float> gamma, beta;
  std::vector<float> g_grad, b_grad;
  float *cache_x_hat;
  float *cache_std_inv;

  LayerNorm(int d_mod, int s_len) : d_model(d_mod), max_seq_len(s_len) {
    gamma.assign(d_model, 1.0f);
    beta.assign(d_model, 0.0f);
    g_grad.assign(d_model, 0.0f);
    b_grad.assign(d_model, 0.0f);
    cache_x_hat = new float[max_seq_len * d_model]();
    cache_std_inv = new float[max_seq_len]();
  }

  ~LayerNorm() {
    delete[] cache_x_hat;
    delete[] cache_std_inv;
  }

  void forward(Tensor &input, Tensor &output) override {
    int L = input.rows;
    int D = input.cols;

    for (int i = 0; i < L; i++) {
      float mean = 0, var = 0;
      float *row_in = &input.data[i * D];
      float *row_xhat = &cache_x_hat[i * D];
      for (int j = 0; j < D; j++)
        mean += row_in[j];
      mean /= D;
      for (int j = 0; j < D; j++) {
        float diff = row_in[j] - mean;
        var += diff * diff;
      }
      var /= D;
      float std_inv = 1.0f / sqrt(var + 1e-6f);
      cache_std_inv[i] = std_inv;

      for (int j = 0; j < D; j++) {
        float x_hat = (row_in[j] - mean) * std_inv;
        row_xhat[j] = x_hat;
        output.data[i * D + j] = x_hat * gamma[j] + beta[j];
      }
    }
  }

  void backward(Tensor &input, Tensor &output) override {
    int L = input.rows;
    int D = input.cols;

    for (int i = 0; i < L; i++) {
      float *dy = &output.grad[i * D];
      float *dx = &input.grad[i * D];
      float *x_hat = &cache_x_hat[i * D];
      float std_inv = cache_std_inv[i];

      float sum_dy = 0;
      float sum_dy_xhat = 0;

      for (int j = 0; j < D; j++) {
        sum_dy += dy[j] * gamma[j];
        sum_dy_xhat += dy[j] * gamma[j] * x_hat[j];
        g_grad[j] += dy[j] * x_hat[j];
        b_grad[j] += dy[j];
      }
      for (int j = 0; j < D; j++) {
        dx[j] += (std_inv / D) *
                 (D * dy[j] * gamma[j] - sum_dy - x_hat[j] * sum_dy_xhat);
      }
    }
  }

  void update(float lr) override {
    for (int i = 0; i < d_model; i++) {
      gamma[i] -= lr * g_grad[i];
      beta[i] -= lr * b_grad[i];
    }
  }

  void clear_grad() override {
    std::fill(g_grad.begin(), g_grad.end(), 0.0f);
    std::fill(b_grad.begin(), b_grad.end(), 0.0f);
  }
};
class PositionalEncoding {
public:
  int max_len;
  int d_model;
  Tensor pe;
  PositionalEncoding(int len, int d_mod)
      : max_len(len), d_model(d_mod), pe(len, d_mod) {
    for (int pos = 0; pos < len; pos++) {
      for (int i = 0; i < d_model; i += 2) {
        float div_term = pow(10000.0f, (float)i / d_model);
        pe.data[pos * d_model + i] = sin(pos / div_term);
        if (i + 1 < d_model) {
          pe.data[pos * d_model + i + 1] = cos(pos / div_term);
        }
      }
    }
  }

  void forward(Tensor &input) {
    for (int i = 0; i < input.rows * input.cols; i++) {
      input.data[i] += pe.data[i];
    }
  }
  void backward(Tensor &output_grad, Tensor &input_grad) {
    for (int i = 0; i < output_grad.rows * output_grad.cols; i++) {
      input_grad.data[i] = output_grad.data[i];
    }
  }
};
void clip_grad(Tensor *t, float limit) {
  for (int i = 0; i < t->rows * t->cols; i++) {
    if (t->grad[i] > limit)
      t->grad[i] = limit;
    if (t->grad[i] < -limit)
      t->grad[i] = -limit;
  }
}

std::string load_corpus(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Connot open corpus at:" << path << std::endl;
    return "";
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}
void save_weights(std::ofstream &out, Tensor &W) {
  if (W.data == nullptr)
    return;
  out.write(reinterpret_cast<char *>(W.data), W.rows * W.cols * sizeof(float));
}

void load_weights(std::ifstream &in, Tensor &W) {
  if (W.data == nullptr)
    return;
  in.read(reinterpret_cast<char *>(W.data), W.rows * W.cols * sizeof(float));
}

void extract_image_patches(const string &path, Tensor &patches,
                           int patch_size = 16, int image_size = 224) {
  if (patch_size <= 0 || image_size <= 0 || image_size % patch_size != 0 ||
      patches.rows != (image_size / patch_size) * (image_size / patch_size) ||
      patches.cols != patch_size * patch_size * 3)
    throw std::invalid_argument("Image patch dimensions are invalid");
  int w, h, channels;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load(path.c_str(), &w, &h, &channels, 3), stbi_image_free);
  if (!pixels) throw std::runtime_error("Cannot load image: " + path);
  const int side = image_size / patch_size;
  for (int patch = 0; patch < patches.rows; ++patch) {
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x) {
        const float sx = static_cast<float>((patch % side) * patch_size + x) * w / image_size;
        const float sy = static_cast<float>((patch / side) * patch_size + y) * h / image_size;
        const int x0 = std::min(static_cast<int>(sx), w - 1);
        const int y0 = std::min(static_cast<int>(sy), h - 1);
        const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float dx = sx - x0, dy = sy - y0;
        for (int c = 0; c < 3; ++c) {
          const auto *data = pixels.get();
          const float value =
              data[(y0 * w + x0) * 3 + c] * (1 - dx) * (1 - dy) +
              data[(y0 * w + x1) * 3 + c] * dx * (1 - dy) +
              data[(y1 * w + x0) * 3 + c] * (1 - dx) * dy +
              data[(y1 * w + x1) * 3 + c] * dx * dy;
          patches.data[patch * patches.cols + (y * patch_size + x) * 3 + c] = value / 255.0f;
        }
      }
    }
  }
}

Tensor image_to_patches(const string &path, int d_model,
                        int patch_size = 16, int image_size = 224) {
  if (patch_size <= 0 || image_size <= 0 || image_size % patch_size != 0)
    throw std::invalid_argument("Image patch dimensions are invalid");
  int side = image_size / patch_size;
  Tensor patches(side * side, patch_size * patch_size * 3);
  extract_image_patches(path, patches, patch_size, image_size);
  LinearLayer projection(patches.cols, d_model);
  Tensor output(patches.rows, d_model);
  projection.forward(patches, output);
  return output;
}
class TransformerBlock {
public:
  LayerNorm norm1;
  AttentionLayer attn;
  LayerNorm norm2;
  LinearLayer ffn1;
  LinearLayer ffn2;

  int d_model;
  int seq_len;

  Tensor attn_norm_cache;
  Tensor attn_out_cache;
  Tensor ffn_norm_cache;
  Tensor ffn_mid_cache;
  Tensor ffn_out_cache;
  Tensor output_cache;
  Tensor grad_cache;

  TransformerBlock(int seq_len, int d_model, int d_head)
      : norm1(d_model, seq_len), attn(seq_len, d_model, d_head),
        norm2(d_model, seq_len), ffn1(d_model, d_model * 4),
        ffn2(d_model * 4, d_model), attn_norm_cache(seq_len, d_model),
        attn_out_cache(seq_len, d_model), ffn_norm_cache(seq_len, d_model),
        ffn_mid_cache(seq_len, d_model * 4), ffn_out_cache(seq_len, d_model),
        output_cache(seq_len, d_model), grad_cache(seq_len, d_model) {
    this->d_model = d_model;
    this->seq_len = seq_len;
  }

  void forward(Tensor &input, Tensor &output) {
    norm1.forward(input, attn_norm_cache);
    attn.forward(attn_norm_cache, attn_out_cache);
    for (int i = 0; i < seq_len * d_model; ++i) {
      attn_out_cache.data[i] += input.data[i];
    }
    norm2.forward(attn_out_cache, ffn_norm_cache);
    ffn1.forward(ffn_norm_cache, ffn_mid_cache);
    for (int i = 0; i < ffn_mid_cache.rows * ffn_mid_cache.cols; ++i) {
      if (ffn_mid_cache.data[i] < 0)
        ffn_mid_cache.data[i] = 0;
    }

    ffn2.forward(ffn_mid_cache, ffn_out_cache);
    for (int i = 0; i < seq_len * d_model; ++i) {
      output.data[i] = ffn_out_cache.data[i] + attn_out_cache.data[i];
    }
    memcpy(output_cache.data, output.data, seq_len * d_model * sizeof(float));
  }

  void backward(Tensor &input, Tensor &output_grad) {
    ffn2.backward(ffn_mid_cache, output_grad);
    for (int i = 0; i < seq_len * d_model * 4; i++) {
      if (ffn_mid_cache.data[i] <= 0)
        ffn_mid_cache.grad[i] = 0;
    }

    ffn1.backward(ffn_norm_cache, ffn_mid_cache);
    norm2.backward(attn_out_cache, ffn_norm_cache);
    for (int i = 0; i < seq_len * d_model; i++) {
      attn_out_cache.grad[i] += output_grad.grad[i];
    }

    attn.backward(attn_norm_cache, attn_out_cache);
    norm1.backward(input, attn_norm_cache);
    for (int i = 0; i < seq_len * d_model; i++) {
      input.grad[i] += attn_out_cache.grad[i];
    }
  }
  void update(float lr) {
    norm1.update(lr);
    norm2.update(lr);
    attn.update(lr);
    ffn1.update(lr);
    ffn2.update(lr);
  }
  void clear_grad() {
    attn.clear_grad();
    ffn1.clear_grad();
    ffn2.clear_grad();
    // 关键：反向传播里大量使用了 `+=` 更新 cache.grad，
    // 如果这里不清零，会导致梯度在 step 间累积而数值爆炸。
    norm1.clear_grad();
    norm2.clear_grad();
    attn_norm_cache.clear_grad();
    attn_out_cache.clear_grad();
    ffn_norm_cache.clear_grad();
    ffn_mid_cache.clear_grad();
    ffn_out_cache.clear_grad();
    output_cache.clear_grad();
    grad_cache.clear_grad();
  }
  void save(ofstream &out) {
    attn.save(out);
    ffn1.save(out);
    ffn2.save(out);
  }
  void load(ifstream &in) {
    attn.load(in);
    ffn1.load(in);
    ffn2.load(in);
  }
};
void global_clip(TransformerBlock *b, float limit) {
  clip_grad(b->ffn1.W, limit);
  clip_grad(b->ffn2.W, limit);
  clip_grad(b->attn.W_q.W, limit);
  clip_grad(b->attn.W_k.W, limit);
  clip_grad(b->attn.W_v.W, limit);
}
void process_image_to_input(const string &path, LinearLayer &projection,
                            Tensor &input, Tensor &patches, Tensor &image_embeddings) {
  if (input.rows < image_embeddings.rows || input.cols != image_embeddings.cols)
    throw std::invalid_argument("Image embeddings do not fit the input tensor");
  extract_image_patches(path, patches);
  projection.forward(patches, image_embeddings);
  std::copy_n(image_embeddings.data, image_embeddings.rows * image_embeddings.cols, input.data);
}

checkpoint::State model_state(EmbeddingLayer &embed, LinearLayer &vision,
                              vector<std::unique_ptr<TransformerBlock>> &blocks,
                              LinearLayer &projection, int seq_len,
                              const bpe::BPETrainer &tokenizer) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Checkpoints require IEEE-754 32-bit floats");
  checkpoint::State state;
  state.metadata = {{"format_version", 1}, {"seq_len", seq_len},
                    {"d_model", embed.d_model}, {"vocab_size", embed.vocab_size},
                    {"block_count", blocks.size()}, {"scalar_format", "ieee754-f32"},
                    {"byte_order", checkpoint::byte_order()},
                    {"tokenizer_fingerprint", tokenizer.fingerprint()},
                    {"vision_layout", "rgb-bilinear-224-p16"}};
  auto tensor = [&](Tensor &value) {
    state.parameters.emplace_back(value.data, static_cast<size_t>(value.rows) * value.cols);
  };
  auto linear = [&](LinearLayer &layer) { tensor(*layer.W); tensor(*layer.b); };
  tensor(embed.weights);
  linear(vision);
  for (auto &block : blocks) {
    linear(block->attn.W_q);
    linear(block->attn.W_k);
    linear(block->attn.W_v);
    linear(block->ffn1);
    linear(block->ffn2);
  }
  linear(projection);
  state.legacy_blocks = state.parameters.size();
  for (auto &block : blocks)
    for (auto *norm : {&block->norm1, &block->norm2}) {
      state.parameters.emplace_back(norm->gamma.data(), norm->gamma.size());
      state.parameters.emplace_back(norm->beta.data(), norm->beta.size());
    }
  return state;
}

int main(int argc, char **argv) {
  try {


  const string config_path = argc > 1 ? argv[1] : "config.json";
  ifstream cfg_file(config_path);
  if (!cfg_file.is_open()) {
    cout << "Error: configuration not found: " << config_path << endl;
    return -1;
  }
  json cfg;
  cfg_file >> cfg;
  srand(cfg["training"].value("seed", static_cast<unsigned int>(time(nullptr))));
  string corpus_path = cfg["training"].value("corpus_path", "train.txt");
  int seq_len = cfg["model"].value("seq_len", 128);
  int d_model = cfg["model"].value("d_model", 128);
  int epochs = cfg["training"].value("epochs", 5000);
  float lr = cfg["training"].value("learning_rate", 0.0001f);
  float clip_threshold = cfg["training"].value("clip_threshold", 1.0f);
  int save_point = cfg["training"].value("save_point", 100);
  string weight_path = cfg["training"].value("save_path", "model.bin");
  string load_path = cfg["training"].value("load_path", "");
  if (seq_len <= 0 || d_model <= 0 ||
      d_model > std::numeric_limits<int>::max() / 4 || epochs <= 0 ||
      !std::isfinite(lr) || lr <= 0 || !std::isfinite(clip_threshold) ||
      clip_threshold <= 0 || save_point < 0)
    throw std::invalid_argument("Model dimensions, epochs, learning rate and clip threshold must be positive; save_point must be nonnegative");
  if (!fs::exists(corpus_path))
    throw std::invalid_argument("Corpus does not exist: " + corpus_path);
  bool is_vlm = fs::is_directory(corpus_path);
  if (is_vlm && seq_len <= 196)
    throw std::invalid_argument("VLM seq_len must exceed the 196 image tokens");
  json vlm_dataset;
  if (is_vlm) {
    ifstream dataset_file(fs::path(corpus_path) / "train.json");
    if (!dataset_file) throw std::runtime_error("Cannot open VLM train.json");
    dataset_file >> vlm_dataset;
    if (!vlm_dataset.is_array() || vlm_dataset.empty())
      throw std::invalid_argument("VLM train.json must contain a nonempty array");
    for (const auto &sample : vlm_dataset)
      if (!sample.is_object() || !sample.contains("answer") ||
          !sample["answer"].is_string() || sample["answer"].get<string>().empty())
        throw std::invalid_argument("Each VLM sample must have a nonempty answer string");
  }
  bpe::BPEConfig bpe_cfg;
  const int requested_vocab_size = cfg["model"].value("vocab_size", 2000);
  if (requested_vocab_size < 260)
    throw std::invalid_argument("vocab_size must be at least 260 for byte-level BPE");
  bpe_cfg.vocab_size = static_cast<size_t>(requested_vocab_size);
  bpe::BPETrainer bpe_model(bpe_cfg);
  string bpe_path = cfg.value("bpe", json::object()).value("bpe_model_path", "bpe_model.bin");
  if (!bpe_model.load(bpe_path)) {
    if (is_vlm) {
      vector<string> answers;
      for (const auto &sample : vlm_dataset) answers.push_back(sample["answer"].get<string>());
      bpe_model.train_from_texts(answers);
    } else if (!bpe_model.train_from_file(corpus_path)) {
      throw std::runtime_error("Cannot train BPE from: " + corpus_path);
    }
    if (!bpe_model.save(bpe_path))
      throw std::runtime_error("Cannot save BPE to: " + bpe_path);
  }
  int vocab_size = bpe_model.vocab_size();
  EmbeddingLayer embed(vocab_size, d_model);
  PositionalEncoding pos_enc(seq_len, d_model);
  LinearLayer vision_proj(768, d_model);
  vector<std::unique_ptr<TransformerBlock>> blocks;

  for (int i = 0; i < 8; i++) {
    blocks.push_back(std::make_unique<TransformerBlock>(seq_len, d_model, d_model));
  }
  LinearLayer projection(d_model, vocab_size);

  const auto checkpoint_state = model_state(embed, vision_proj, blocks, projection, seq_len, bpe_model);
  const string effective_load_path = load_path.empty() ? weight_path : load_path;
  if (!load_path.empty() || fs::exists(effective_load_path)) {
    cout << "[System] Loading weights..." << endl;
    if (checkpoint::load(effective_load_path, checkpoint_state))
      cout << "[Warn] Legacy checkpoint: dimensions/tokenizer cannot be verified; LayerNorm starts at defaults. Next save uses format version 1." << endl;
  }
  vector<bpe::TokenId> all_text_tokens;
  if (!is_vlm) {
    ifstream t_in(corpus_path);
    string line;
    while (getline(t_in, line)) {
      auto lt = bpe_model.encode(line, false);
      all_text_tokens.insert(all_text_tokens.end(), lt.begin(), lt.end());
    }
  }
  if (!is_vlm && all_text_tokens.size() < static_cast<size_t>(seq_len) + 1)
    throw std::invalid_argument("Text corpus must contain at least seq_len + 1 encoded tokens");
  Tensor input_tensor(seq_len, d_model);
  Tensor hidden(seq_len, d_model);
  Tensor next_h(seq_len, d_model);
  Tensor logits(seq_len, vocab_size);

  std::unique_ptr<Tensor> image_patches, image_embeddings;
  if (is_vlm) {
    image_patches = std::make_unique<Tensor>(196, 768);
    image_embeddings = std::make_unique<Tensor>(196, d_model);
  }
  auto save_model = [&]() { checkpoint::save(weight_path, checkpoint_state); };

  cout << "[System] Starting " << (is_vlm ? "VLM" : "Text")
       << " training loop..." << endl;

  // 自适应学习率状态：基于 loss EMA 的 plateau 检测
  float loss_ema = -1.0f;
  float best_loss_ema = 1e30f;
  float lr_scale = 1.0f;
  int plateau_count = 0;
  int lr_cooldown = 0;
  const float lr_scale_min = 0.5f;
  const float lr_scale_max = 2.0f;
  const int plateau_patience = 80;
  const int cooldown_steps = 60;
  const float rel_improve_eps = 0.003f; // 0.3% 才算显著改进

  for (int epoch = 0; epoch < epochs; epoch++) {
    auto reset_gradients = [&]() {
      embed.clear_grad();
      projection.clear_grad();
      input_tensor.clear_grad();
      hidden.clear_grad();
      next_h.clear_grad();
      logits.clear_grad();
      if (is_vlm)
        vision_proj.clear_grad();
      for (auto &b : blocks)
        b->clear_grad();
    };
    reset_gradients();

    // -1 表示该位置不参与监督；0 是合法 token id，不能再用 0 代表 ignore
    vector<int> target_ids(seq_len, -1);
    vector<int> step_input_ids(seq_len, -1);

    if (is_vlm) {
      int idx = rand() % (int)vlm_dataset.size();
      string img_p =
          (fs::path(corpus_path) / ("train_" + to_string(idx) + ".jpg"))
              .string();
      const int img_tokens = 196;
      // 保底：VLM 分支可能不会填满整个 seq_len，必须清空未写入区域
      std::memset(input_tensor.data, 0,
                  input_tensor.rows * input_tensor.cols * sizeof(float));
      process_image_to_input(img_p, vision_proj, input_tensor, *image_patches, *image_embeddings);
      string ans = vlm_dataset[idx].value("answer", "");
      auto tokens = bpe_model.encode(ans, true);
      for (size_t i = 0; i + 1 < tokens.size() &&
                         (i + img_tokens < (size_t)seq_len);
           i++) {
        int current_id = tokens[i];
        int next_id = tokens[i + 1];
        if (current_id < 0 || current_id >= vocab_size || next_id < 0 ||
            next_id >= vocab_size) {
          continue;
        }
        float *dest = &input_tensor.data[(i + img_tokens) * d_model];
        float *src = &embed.weights.data[current_id * d_model];
        memcpy(dest, src, sizeof(float) * d_model);
        step_input_ids[i + img_tokens] = current_id;
        target_ids[i + img_tokens] = next_id;
      }
    } else {
      size_t start = static_cast<size_t>(rand()) % (all_text_tokens.size() - seq_len);
      vector<int> ids;
      for (int i = 0; i < seq_len; i++) {
        ids.push_back((int)all_text_tokens[start + i]);
        target_ids[i] = (int)all_text_tokens[start + i + 1];
        step_input_ids[i] = ids.back();
      }
      embed.forward_ids(ids, input_tensor);
    }

    pos_enc.forward(input_tensor);
    memcpy(hidden.data, input_tensor.data, seq_len * d_model * sizeof(float));

    for (int i = 0; i < 8; i++) {
      blocks[i]->forward(hidden, next_h);
      memcpy(hidden.data, next_h.data, seq_len * d_model * sizeof(float));
    }
    projection.forward(hidden, logits);

    float total_loss = 0;
    int count = 0;
    memset(logits.grad, 0, seq_len * vocab_size * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
      int target = target_ids[t];
      if (target < 0 || target >= vocab_size)
        continue;

      float max_v = -1e9f;
      float *t_logits = &logits.data[t * vocab_size];
      for (int v = 0; v < vocab_size; v++) {
        if (!std::isfinite(t_logits[v]))
          t_logits[v] = 0.0f;
        if (t_logits[v] > max_v)
          max_v = t_logits[v];
      }

      float sum_exp = 0;
      for (int v = 0; v < vocab_size; v++) {
        float val = exp(t_logits[v] - max_v);
        if (!std::isfinite(val))
          val = 0.0f;
        logits.grad[t * vocab_size + v] = val;
        sum_exp += val;
      }
      if (!std::isfinite(sum_exp) || sum_exp <= 0.0f)
        continue;

      float prob =
          (logits.grad[t * vocab_size + target]) / (sum_exp + 1e-10f);
      total_loss -= log(prob + 1e-10f);
      count++;

      for (int v = 0; v < vocab_size; v++) {
        float p = logits.grad[t * vocab_size + v] / (sum_exp + 1e-10f);
        logits.grad[t * vocab_size + v] = p - (v == target ? 1.0f : 0.0f);
      }
    }

    if (count == 0) continue;
    for (int i = 0; i < seq_len * vocab_size; ++i)
      logits.grad[i] /= static_cast<float>(count);
    projection.backward(hidden, logits);
    for (int i = 7; i >= 0; i--) {
      Tensor &input_ref =
          (i == 0) ? input_tensor : blocks[i - 1]->output_cache;
      blocks[i]->backward(input_ref, hidden);
      if (i > 0)
        memcpy(hidden.grad, input_ref.grad, seq_len * d_model * sizeof(float));
    }

    if (!is_vlm) {
      embed.backward(input_tensor, input_tensor);
    } else {
      std::copy_n(input_tensor.grad, 196 * d_model, image_embeddings->grad);
      vision_proj.backward(*image_patches, *image_embeddings);
      // VLM: 仅对文本 token 位置把输入梯度回传到词向量。
      for (int pos = 0; pos < seq_len; pos++) {
        int id = step_input_ids[pos];
        if (id < 0 || id >= vocab_size)
          continue;
        for (int d = 0; d < d_model; d++) {
          embed.weights.grad[id * d_model + d] +=
              input_tensor.grad[pos * d_model + d];
        }
      }
    }

    auto strict_clip = [&](Tensor *t) {
      if (!t || !t->grad)
        return;
      for (int i = 0; i < t->rows * t->cols; i++) {
        if (t->grad[i] > clip_threshold)
          t->grad[i] = clip_threshold;
        if (t->grad[i] < -clip_threshold)
          t->grad[i] = -clip_threshold;
      }
    };
    auto clip_linear = [&](LinearLayer &layer) {
      strict_clip(layer.W);
      strict_clip(layer.b);
    };
    auto clip_norm = [&](LayerNorm &norm) {
      for (auto *gradient : {&norm.g_grad, &norm.b_grad})
        for (float &value : *gradient)
          value = std::clamp(value, -clip_threshold, clip_threshold);
    };
    clip_linear(projection);
    strict_clip(&embed.weights);
    if (is_vlm)
      clip_linear(vision_proj);
    for (auto &b : blocks) {
      clip_linear(b->ffn1);
      clip_linear(b->ffn2);
      clip_linear(b->attn.W_q);
      clip_linear(b->attn.W_k);
      clip_linear(b->attn.W_v);
      clip_norm(b->norm1);
      clip_norm(b->norm2);
    }

    // 前 5% 线性 warmup，后续由自适应调度主导（避免后期 lr 过小）
    int warmup_steps = std::max(1, epochs / 20);
    float scheduled_lr = lr;
    if (epoch < warmup_steps) {
      scheduled_lr = lr * (float)(epoch + 1) / (float)warmup_steps;
    }
    float current_lr = scheduled_lr * lr_scale;

    projection.update(current_lr);
    for (auto &b : blocks)
      b->update(current_lr);
    if (is_vlm)
      vision_proj.update(current_lr);
    embed.update(current_lr);

    float avg_loss = (count > 0 ? total_loss / count : 0.0f);
    if (count > 0 && std::isfinite(avg_loss)) {
      if (loss_ema < 0.0f)
        loss_ema = avg_loss;
      else
        loss_ema = 0.95f * loss_ema + 0.05f * avg_loss;

      float improve_ratio = (best_loss_ema - loss_ema) /
                            std::max(1e-6f, std::fabs(best_loss_ema));
      if (improve_ratio > rel_improve_eps) {
        best_loss_ema = loss_ema;
        plateau_count = 0;
        // 持续变好时，允许轻微回升 lr（避免过早降得太低）
        lr_scale = std::min(lr_scale_max, lr_scale * 1.008f);
      } else {
        plateau_count++;
        if (lr_cooldown > 0) {
          lr_cooldown--;
        } else if (plateau_count >= plateau_patience) {
          float old_scale = lr_scale;
          lr_scale = std::max(lr_scale_min, lr_scale * 0.9f);
          plateau_count = 0;
          lr_cooldown = cooldown_steps;
          if (lr_scale < old_scale - 1e-8f) {
            cout << "[LR] plateau detected, lr_scale -> " << lr_scale << endl;
          }
        }
      }
    }

    if (epoch % 10 == 0) {
      printf("Epoch %d | Loss: %.4f | EMA: %.4f | LR: %.6f | Scale: %.3f\n",
             epoch, avg_loss, (loss_ema > 0 ? loss_ema : avg_loss), current_lr,
             lr_scale);
    }

    if (save_point > 0 && (epoch + 1) % save_point == 0) {
      save_model();
      cout << "[System] Checkpoint saved (epoch " << (epoch + 1) << ")" << endl;
    }
  }

  save_model();
  cout << "[System] Model saved" << endl;
  return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}