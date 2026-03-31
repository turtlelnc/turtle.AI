#include "BPE.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
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
#include <omp.h>
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
    data = new float[r * c];
    grad = new float[r * c];
  }
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&other) noexcept
      : rows(other.rows), cols(other.cols), data(other.data), grad(other.grad) {
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
memset(c, 0, m * n * sizeof(float));
#pragma omp parallel for collapse(2)
for (int i = 0; i < m; i++) {
  for (int f = 0; f < k; f++) {
    float a_if = a[i * k + f];
    for (int j = 0; j < n; j++) {
      c[i * n + j] += a_if * b[f * n + j]; // 内存连续访问！
    }
  }
}
#endif
}
void linear_forward(Tensor &X, Tensor &W, Tensor &Y) {
  matmul(X.data, W.data, Y.data, X.rows, X.cols, W.cols);
}
void linear_backward(Tensor &X, Tensor &W, Tensor &Y) {
  Tensor x_tp(X.rows, X.cols);
  transpose(X.data, x_tp.data, X.rows, X.cols);
  matmul(x_tp.data, Y.grad, W.grad, x_tp.rows, x_tp.cols, Y.cols);
  Tensor w_tp(X.rows, X.cols);
  transpose(X.data, w_tp.data, W.rows, W.cols);
  matmul(w_tp.data, Y.grad, W.grad, Y.rows, Y.cols, w_tp.cols);
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
    Y_pred.grad[i] = Y_pred.data[i] - Y_target.data[i];
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
  float *x_tp_buffer;
  float *w_tp_buffer;
  int max_batch_size;

  LinearLayer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim) {
    W = new Tensor(in_dim, out_dim);
    b = new Tensor(1, out_dim);
    max_batch_size = 2048;
    x_tp_buffer = new float[max_batch_size * in_dim]();
    w_tp_buffer = new float[in_dim * out_dim]();

    float scale = sqrt(2.0f / in_dim);
    for (int i = 0; i < W->rows * W->cols; i++) {
      W->data[i] = ((rand() / (float)RAND_MAX) - 0.5f) * scale;
    }
    for (int i = 0; i < b->cols; i++)
      b->data[i] = 0.0f;
  }

  ~LinearLayer() {
    delete W;
    delete b;
    delete[] x_tp_buffer;
    delete[] w_tp_buffer;
  }

  void forward(Tensor &input, Tensor &output) override {
    matmul(input.data, W->data, output.data, input.rows, input.cols, W->cols);

#if !USE_ACCELERATE
#pragma omp parallel for
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

    if (M > max_batch_size) {
      printf("[Error] Batch size %d exceeds max %d\n", M, max_batch_size);
      return;
    }

    float *x_tp_data = x_tp_buffer;
    float *w_tp_data = w_tp_buffer;

    transpose(input.data, x_tp_data, M, K);
    matmul(x_tp_data, output.grad, W->grad, K, M, N);

    transpose(W->data, w_tp_data, K, N);
    matmul(output.grad, w_tp_data, input.grad, M, N, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        b->grad[j] += output.grad[i * N + j];
      }
    }
  }

  void save(std::ofstream &out) {
    W->save(out);
    b->save(out);
  }

  void load(std::ifstream &in) {
    W->load(in);
    b->load(in);
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
  Tensor K_tp, V_tp;
  Tensor scores, scores_softmax;
  Tensor d_scores_softmax, d_scores_raw, d_scores_raw_tp;
  Tensor scores_softmax_tp;

  AttentionLayer(int s, int m, int h)
      : W_q(m, h), W_k(m, h), W_v(m, h), seq_len(s), d_model(m), d_head(h),
        Q(s, h), K(s, h), V(s, h), K_tp(h, s), V_tp(h, s), scores(s, s),
        scores_softmax(s, s), d_scores_softmax(s, s), d_scores_raw(s, s),
        d_scores_raw_tp(s, s), scores_softmax_tp(s, s) {}

  void forward(Tensor &input, Tensor &output, int batch_size) override {
      // 1. 线性层：由于 input 现在的 rows 是 batch_size * seq_len
      // W_q, W_k, W_v 的 forward 会自动一次性算出整个 Batch 的所有 Token 的 QKV
      // 这一步利用了大矩阵乘法，性能极高
      W_q.forward(input, Q); // Q 维度: [batch_size * seq_len, d_head]
      W_k.forward(input, K);
      W_v.forward(input, V);
  
      // 2. 注意力计算：这一步必须通过循环来隔离不同的 Batch
      #pragma omp parallel for schedule(dynamic)
      for (int b = 0; b < batch_size; b++) {
          // 计算每个 Batch 在连续内存中的起始指针偏移
          int qkv_offset = b * seq_len * d_head;
          int score_offset = b * seq_len * seq_len;
  
          float* b_Q = &Q.data[qkv_offset];
          float* b_K = &K.data[qkv_offset];
          float* b_V = &V.data[qkv_offset];
          float* b_scores = &scores.data[score_offset];
          float* b_scores_sm = &scores_softmax.data[score_offset];
          float* b_output = &output.data[qkv_offset];
  
          // 临时转置矩阵需要是 thread_local 或者每个 Batch 独立，防止冲突
          // 建议在 AttentionLayer 里预分配 K_tp 为 [batch_size * d_head * seq_len]
          float* b_K_tp = &K_tp.data[b * d_head * seq_len];
  
          // --- 以下是原有的单样本逻辑，搬进循环里 ---
          transpose(b_K, b_K_tp, seq_len, d_head);
          matmul(b_Q, b_K_tp, b_scores, seq_len, d_head, seq_len);
  
          float scale = 1.0f / sqrt((float)d_head);
          for (int i = 0; i < seq_len; i++) {
              for (int j = 0; j < seq_len; j++) {
                  if (j > i) {
                      b_scores[i * seq_len + j] = -1e9f; // Mask
                  } else {
                      b_scores[i * seq_len + j] *= scale;
                  }
              }
              softmax(&b_scores[i * seq_len], seq_len);
              memcpy(&b_scores_sm[i * seq_len], &b_scores[i * seq_len], sizeof(float) * seq_len);
          }
          
          // Output = Score * V
          matmul(b_scores_sm, b_V, b_output, seq_len, seq_len, d_head);
      }
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
    last_input_ids = token_ids;
    for (int i = 0; i < (int)token_ids.size(); i++) {
      int id = token_ids[i];
      if (id >= vocab_size)
        id = 0;
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

Tensor image_to_patches(const string &img_path, int d_model,
                        int patch_size = 16, int image_size = 224) {
  int w, h, c;
  unsigned char *data = stbi_load(img_path.c_str(), &w, &h, &c, 3);
  if (!data) {
    printf("[Vision] Failed to load image: %s\n", img_path.c_str());
    return Tensor(0, d_model);
  }
  int target_h = image_size, target_w = image_size;
  vector<float> resized(target_h * target_w * 3);

  for (int y = 0; y < target_h; y++) {
    for (int x = 0; x < target_w; x++) {
      float src_x = (float)x * w / target_w;
      float src_y = (float)y * h / target_h;
      int x0 = min((int)src_x, w - 2), y0 = min((int)src_y, h - 2);
      float dx = src_x - x0, dy = src_y - y0;

      for (int ch = 0; ch < 3; ch++) {
        float v00 = data[(y0 * w + x0) * 3 + ch];
        float v01 = data[(y0 * w + x0 + 1) * 3 + ch];
        float v10 = data[((y0 + 1) * w + x0) * 3 + ch];
        float v11 = data[((y0 + 1) * w + x0 + 1) * 3 + ch];
        float val = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) +
                    v10 * (1 - dx) * dy + v11 * dx * dy;
        resized[(y * target_w + x) * 3 + ch] = val / 255.0f;
      }
    }
  }
  stbi_image_free(data);

  int num_patches = (image_size / patch_size) * (image_size / patch_size);
  int patch_dim = patch_size * patch_size * 3;

  Tensor patch_embed(num_patches, d_model);
  vector<float> proj(patch_dim * d_model);
  for (auto &v : proj)
    v = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

  for (int p = 0; p < num_patches; p++) {
    int py = (p / (image_size / patch_size)) * patch_size;
    int px = (p % (image_size / patch_size)) * patch_size;

    vector<float> patch_vec(patch_dim);
    for (int ky = 0; ky < patch_size; ky++) {
      for (int kx = 0; kx < patch_size; kx++) {
        for (int ch = 0; ch < 3; ch++) {
          int idx = ((py + ky) * target_w + (px + kx)) * 3 + ch;
          patch_vec[(ky * patch_size + kx) * 3 + ch] = resized[idx];
        }
      }
    }
    for (int d = 0; d < d_model; d++) {
      float sum = 0;
      for (int i = 0; i < patch_dim; i++) {
        sum += patch_vec[i] * proj[i * d_model + d];
      }
      patch_embed.data[p * d_model + d] = sum;
    }
  }

  printf("[Vision] Image %s -> %d patches (%d-dim)\n", img_path.c_str(),
         num_patches, d_model);
  return patch_embed;
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
#include <cstring>
void process_image_to_input(const string &path, LinearLayer &vis_proj,
                            Tensor &input_tensor, int d_model) {
  int w, h, c;
  unsigned char *data = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!data)
  {
    // 保底：避免未初始化 data 进入后续计算导致 NaN
    std::memset(input_tensor.data, 0,
                input_tensor.rows * input_tensor.cols * sizeof(float));
    return;
  }

  int patch_size = 16;
  int num_patches_side = 14;
  int patch_dim = 768;

  for (int i = 0; i < num_patches_side * num_patches_side; i++) {
    if (i >= input_tensor.rows)
      break;
    float patch_raw[768];
    for (int j = 0; j < 768; j++) {
      patch_raw[j] = (float)data[(i * 768 + j) % (w * h * c)] / 255.0f;
    }
    for (int d = 0; d < d_model; d++) {
      float sum = 0;
      for (int p = 0; p < patch_dim; p++) {
        sum += patch_raw[p] * vis_proj.W->data[p * d_model + d];
      }
      input_tensor.data[i * d_model + d] = sum + vis_proj.b->data[d];
    }
  }
  stbi_image_free(data);
}
  struct VLMSample {
      std::vector<float> img_features;
      std::vector<int> answer_tokens; 
  };
int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    // 1. 加载配置
    ifstream cfg_file("config.json");
    if (!cfg_file.is_open()) {
        cout << "Error: config.json not found!" << endl;
        return -1;
    }
    json cfg;
    cfg_file >> cfg;

    string corpus_path = cfg["training"].value("corpus_path", "train.txt");
    int seq_len = cfg["model"].value("seq_len", 128);
    int d_model = cfg["model"].value("d_model", 128);
    int epochs = cfg["training"].value("epochs", 5000);
    float lr = cfg["training"].value("learning_rate", 0.001f);
    int batch_size = cfg["training"].value("batch_size", 32);
    float clip_threshold = cfg["training"].value("clip_threshold", 1.0f);
    int save_point = cfg["training"].value("save_point", 100);
    string weight_path = cfg["training"].value("save_path", "model.bin");
    string load_path = cfg["training"].value("load_path", "");

    bool is_vlm = fs::is_directory(corpus_path);
    int img_tokens = 196; // 14x14 patches

    // 2. 初始化模型组件
    bpe::BPEConfig bpe_cfg;
    bpe_cfg.vocab_size = cfg["model"].value("vocab_size", 2000);
    bpe::BPETrainer bpe_model(bpe_cfg);
    string bpe_path = cfg["bpe"].value("bpe_model_path", "bpe_model.bin");
    
    if (!bpe_model.load(bpe_path)) {
        string src = is_vlm ? (fs::path(corpus_path) / "train.json").string() : corpus_path;
        bpe_model.train_from_file(src);
        bpe_model.save(bpe_path);
    }

    int vocab_size = bpe_model.vocab_size();
    EmbeddingLayer embed(vocab_size, d_model);
    PositionalEncoding pos_enc(seq_len, d_model);
    LinearLayer vision_proj(768, d_model);
    LinearLayer projection(d_model, vocab_size);

    vector<TransformerBlock *> blocks;
    for (int i = 0; i < 8; i++) {
        // 注意：Block 内部缓存需要容纳 BatchSize * SeqLen
        blocks.push_back(new TransformerBlock(batch_size * seq_len, d_model, d_model));
    }

    // --- 权重加载与检查逻辑 ---
    auto tensor_all_finite = [&](Tensor &t) {
        for (int i = 0; i < t.rows * t.cols; i++) if (!std::isfinite(t.data[i])) return false;
        return true;
    };
    
    string effective_load_path = load_path.empty() ? weight_path : load_path;
    ifstream in_f(effective_load_path, ios::binary);
    if (in_f.is_open()) {
        cout << "[System] Loading weights..." << endl;
        embed.weights.load(in_f);
        vision_proj.load(in_f);
        for (auto b : blocks) b->load(in_f);
        projection.load(in_f);
        in_f.close();
    }

    // 3. 预处理数据 (VLM 缓存 或 文本 Token)
    vector<VLMSample> vlm_cache;
    vector<bpe::TokenId> all_text_tokens;

    if (is_vlm) {
        ifstream j_in(fs::path(corpus_path) / "train.json");
        json vlm_dataset;
        j_in >> vlm_dataset;
        cout << "[System] Pre-processing VLM dataset..." << endl;
        for (auto& item : vlm_dataset) {
            VLMSample sample;
            string img_p = (fs::path(corpus_path) / item.value("image", "")).string();
            Tensor temp_img(img_tokens, 768); // 假设原始视觉特征是 768 维
            process_image_to_input(img_p, vision_proj, temp_img, d_model); 
            // 缓存投影后的特征
            sample.img_features.assign(temp_img.data, temp_img.data + img_tokens * d_model);
            sample.answer_tokens = bpe_model.encode(item.value("answer", ""), false);
            vlm_cache.push_back(std::move(sample));
        }
    } else {
        string bin_path = corpus_path + ".tokens.bin";
        ifstream bin_in(bin_path, ios::binary);
        if (bin_in.is_open()) {
            bin_in.seekg(0, ios::end);
            size_t size = bin_in.tellg();
            bin_in.seekg(0, ios::beg);
            all_text_tokens.resize(size / sizeof(bpe::TokenId));
            bin_in.read(reinterpret_cast<char*>(all_text_tokens.data()), size);
        } else {
            cout << "[System] Tokenizing text corpus..." << endl;
            ifstream t_in(corpus_path);
            string line;
            while (getline(t_in, line)) {
                auto lt = bpe_model.encode(line, false);
                all_text_tokens.insert(all_text_tokens.end(), lt.begin(), lt.end());
            }
        }
    }

    // 4. 初始化 Batch 训练需要的 Tensor
    int batch_rows = batch_size * seq_len;
    Tensor input_tensor(batch_rows, d_model);
    Tensor hidden(batch_rows, d_model);
    Tensor next_h(batch_rows, d_model);
    Tensor logits(batch_rows, vocab_size);
    vector<int> all_target_ids(batch_rows, -1);
    vector<int> all_input_ids(batch_rows, -1);

    // 学习率状态
    float loss_ema = -1.0f;
    float best_loss_ema = 1e30f;
    float lr_scale = 1.0f;
    int plateau_count = 0;

    cout << "[System] Starting " << (is_vlm ? "VLM" : "Text") << " training loop..." << endl;

    // 5. 训练主循环
    for (int epoch = 0; epoch < epochs; epoch++) {
        // --- A. 重置 ---
        embed.clear_grad();
        projection.clear_grad();
        if (is_vlm) vision_proj.clear_grad();
        for (auto b : blocks) b->clear_grad();
        
        memset(input_tensor.data, 0, batch_rows * d_model * sizeof(float));
        std::fill(all_target_ids.begin(), all_target_ids.end(), -1);
        std::fill(all_input_ids.begin(), all_input_ids.end(), -1);

        // --- B. 填充 Batch 数据 ---
        for (int b = 0; b < batch_size; b++) {
            int row_offset = b * seq_len;
            if (is_vlm) {
                const auto& sample = vlm_cache[rand() % vlm_cache.size()];
                // 填充图片
                int actual_img = std::min(img_tokens, seq_len);
                memcpy(&input_tensor.data[row_offset * d_model], sample.img_features.data(), actual_img * d_model * sizeof(float));
                // 填充文本
                for (size_t i = 0; i + 1 < sample.answer_tokens.size() && (i + actual_img < (size_t)seq_len - 1); i++) {
                    int cur = sample.answer_tokens[i];
                    int nxt = sample.answer_tokens[i+1];
                    int row = row_offset + i + actual_img;
                    memcpy(&input_tensor.data[row * d_model], &embed.weights.data[cur * d_model], sizeof(float) * d_model);
                    all_target_ids[row] = nxt;
                    all_input_ids[row] = cur;
                }
            } else {
                int start = rand() % (all_text_tokens.size() - seq_len - 1);
                for (int i = 0; i < seq_len; i++) {
                    int cur = all_text_tokens[start + i];
                    int nxt = all_text_tokens[start + i + 1];
                    int row = row_offset + i;
                    memcpy(&input_tensor.data[row * d_model], &embed.weights.data[cur * d_model], sizeof(float) * d_model);
                    all_target_ids[row] = nxt;
                    all_input_ids[row] = cur;
                }
            }
        }

        // --- C. 前向传播 ---
        // 应用位置编码（按样本独立）
        for(int b=0; b<batch_size; b++) {
            Tensor view(seq_len, d_model);
            view.data = &input_tensor.data[b * seq_len * d_model];
            pos_enc.forward(view);
        }

        memcpy(hidden.data, input_tensor.data, batch_rows * d_model * sizeof(float));
        for (auto b_layer : blocks) {
            b_layer->forward(hidden, next_h, batch_size);
            memcpy(hidden.data, next_h.data, batch_rows * d_model * sizeof(float));
        }
        projection.forward(hidden, logits);

        // --- D. 计算 Loss 和 Logits 梯度 ---
        float batch_accum_loss = 0;
        int valid_tokens = 0;
        memset(logits.grad, 0, batch_rows * vocab_size * sizeof(float));

        for (int i = 0; i < batch_rows; i++) {
            int target = all_target_ids[i];
            if (target < 0 || target >= vocab_size) continue;

            float *row_logits = &logits.data[i * vocab_size];
            float *row_grads = &logits.grad[i * vocab_size];
            
            float max_v = -1e9f;
            for(int v=0; v<vocab_size; v++) if(row_logits[v] > max_v) max_v = row_logits[v];
            
            float sum_exp = 0;
            for(int v=0; v<vocab_size; v++) {
                row_grads[v] = expf(row_logits[v] - max_v);
                sum_exp += row_grads[v];
            }
            
            float prob = row_grads[target] / (sum_exp + 1e-10f);
            batch_accum_loss -= logf(prob + 1e-10f);
            
            for(int v=0; v<vocab_size; v++) {
                row_grads[v] = (row_grads[v] / (sum_exp + 1e-10f)) - (v == target ? 1.0f : 0.0f);
            }
            valid_tokens++;
        }

        // --- E. 反向传播 ---
        projection.backward(hidden, logits);
        memcpy(hidden.grad, projection.input_grad, batch_rows * d_model * sizeof(float));

        for (int i = 7; i >= 0; i--) {
            Tensor &input_ref = (i == 0) ? input_tensor : blocks[i-1]->output_cache;
            blocks[i]->backward(input_ref, hidden, batch_size);
            if (i > 0) memcpy(hidden.grad, input_ref.grad, batch_rows * d_model * sizeof(float));
        }

        // 词向量梯度回传
        for (int i = 0; i < batch_rows; i++) {
            int id = all_input_ids[i];
            if (id >= 0) {
                for (int d = 0; d < d_model; d++) 
                    embed.weights.grad[id * d_model + d] += input_tensor.grad[i * d_model + d];
            }
        }

        // --- F. 参数更新 (含梯度裁剪) ---
        float current_lr = lr * lr_scale;
        float update_step = current_lr / (float)batch_size;

        projection.update(update_step);
        for (auto b : blocks) b->update(update_step);
        embed.update(update_step);
        if (is_vlm) vision_proj.update(update_step);

        // --- G. 打印进度与自动调优 ---
        float avg_loss = valid_tokens > 0 ? batch_accum_loss / valid_tokens : 0;
        if (loss_ema < 0) loss_ema = avg_loss;
        else loss_ema = 0.95f * loss_ema + 0.05f * avg_loss;

        if (epoch % 10 == 0) {
            printf("Epoch %d | Loss: %.4f | EMA: %.4f | Scale: %.3f\n", epoch, avg_loss, loss_ema, lr_scale);
        }

        // 简易 Plateau 检测
        if (loss_ema < best_loss_ema) {
            best_loss_ema = loss_ema;
            plateau_count = 0;
        } else {
            plateau_count++;
            if (plateau_count > 200) {
                lr_scale *= 0.5f;
                plateau_count = 0;
                cout << "[LR] Step down to " << lr_scale << endl;
            }
        }

        // 保存模型
        if (save_point > 0 && (epoch + 1) % save_point == 0) {
            ofstream out_f(weight_path, ios::binary);
            embed.weights.save(out_f);
            vision_proj.save(out_f);
            for (auto b : blocks) b->save(out_f);
            projection.save(out_f);
            cout << "[System] Saved checkpoint." << endl;
        }
    }

    for (auto b : blocks) delete b;
    return 0;
}
