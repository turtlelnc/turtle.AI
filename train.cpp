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
#pragma omp parallel for collapse(2)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int f = 0; f < k; f++) {
        sum += a[i * k + f] * b[f * n + j];
      }
      c[i * n + j] = sum;
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

int main() {


  srand(static_cast<unsigned int>(time(NULL)));
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
  float lr = cfg["training"].value("learning_rate", 0.001f);  // 提高默认值到 1e-3
  int batch_size = cfg["training"].value("batch_size", 32);   // 新增 batch_size 配置
  float clip_threshold = cfg["training"].value("clip_threshold", 1.0f);
  int save_point = cfg["training"].value("save_point", 100);
  string weight_path = cfg["training"].value("save_path", "model.bin");
  string load_path = cfg["training"].value("load_path", "");
  bool is_vlm = fs::is_directory(corpus_path);
  bpe::BPEConfig bpe_cfg;
  bpe_cfg.vocab_size = cfg["model"].value("vocab_size", 2000);
  bpe::BPETrainer bpe_model(bpe_cfg);
  string bpe_path = cfg["bpe"].value("bpe_model_path", "bpe_model.bin");
  if (!bpe_model.load(bpe_path)) {
    string src =
        is_vlm ? (fs::path(corpus_path) / "train.json").string() : corpus_path;
    bpe_model.train_from_file(src);
    bpe_model.save(bpe_path);
  }
  int vocab_size = bpe_model.vocab_size();
  EmbeddingLayer embed(vocab_size, d_model);
  PositionalEncoding pos_enc(seq_len, d_model);
  LinearLayer vision_proj(768, d_model);
  vector<TransformerBlock *> blocks;

  for (int i = 0; i < 8; i++) {
    blocks.push_back(new TransformerBlock(seq_len, d_model, d_model));
  }
  LinearLayer projection(d_model, vocab_size);

  auto tensor_all_finite = [&](Tensor &t) {
    for (int i = 0; i < t.rows * t.cols; i++) {
      if (!std::isfinite(t.data[i]))
        return false;
    }
    return true;
  };
  auto reset_tensor_rand = [&](Tensor &t, float scale) {
    for (int i = 0; i < t.rows * t.cols; i++) {
      t.data[i] = ((rand() / (float)RAND_MAX) - 0.5f) * scale;
    }
  };
  auto reset_linear = [&](LinearLayer &l, float scale_w) {
    reset_tensor_rand(*l.W, scale_w);
    for (int i = 0; i < l.b->cols; i++)
      l.b->data[i] = 0.0f;
  };

  string effective_load_path = load_path.empty() ? weight_path : load_path;
  ifstream in_f(effective_load_path, ios::binary);
  if (in_f.is_open()) {
    cout << "[System] Loading weights..." << endl;
    embed.weights.load(in_f);
    vision_proj.load(in_f);
    for (auto b : blocks)
      b->load(in_f);
    projection.load(in_f);
    in_f.close();

    bool bad_weight = !tensor_all_finite(embed.weights) ||
                      !tensor_all_finite(*vision_proj.W) ||
                      !tensor_all_finite(*projection.W);
    for (auto b : blocks) {
      bad_weight = bad_weight || !tensor_all_finite(*b->ffn1.W) ||
                   !tensor_all_finite(*b->ffn2.W) ||
                   !tensor_all_finite(*b->attn.W_q.W) ||
                   !tensor_all_finite(*b->attn.W_k.W) ||
                   !tensor_all_finite(*b->attn.W_v.W);
    }
    if (bad_weight) {
      cout << "[Warn] Loaded weights contain NaN/Inf, reinitializing model."
           << endl;
      reset_tensor_rand(embed.weights, sqrt(2.0f / d_model));
      reset_linear(vision_proj, sqrt(2.0f / 768.0f));
      for (auto b : blocks) {
        reset_linear(b->ffn1, sqrt(2.0f / d_model));
        reset_linear(b->ffn2, sqrt(2.0f / (d_model * 4.0f)));
        reset_linear(b->attn.W_q, sqrt(2.0f / d_model));
        reset_linear(b->attn.W_k, sqrt(2.0f / d_model));
        reset_linear(b->attn.W_v, sqrt(2.0f / d_model));
      }
      reset_linear(projection, sqrt(2.0f / d_model));
    }
  }
  json vlm_dataset;
  vector<bpe::TokenId> all_text_tokens;
  if (is_vlm) {
    ifstream j_in(fs::path(corpus_path) / "train.json");
    if (j_in.is_open())
      j_in >> vlm_dataset;
  } else {
    ifstream t_in(corpus_path);
    string line;
    while (getline(t_in, line)) {
      auto lt = bpe_model.encode(line, false);
      all_text_tokens.insert(all_text_tokens.end(), lt.begin(), lt.end());
    }
  }
  Tensor input_tensor(seq_len, d_model);
  Tensor hidden(seq_len, d_model);
  Tensor next_h(seq_len, d_model);
  Tensor logits(seq_len, vocab_size);

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
  const int plateau_patience = 150;  // 从 80 增加到 150，避免过早降 LR
  const int cooldown_steps = 60;
  const float rel_improve_eps = 0.002f; // 从 0.003 降到 0.002，更容易触发改进判定

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
      for (auto b : blocks)
        b->clear_grad();
    };
    reset_gradients();

    // 累积梯度用于模拟更大的 batch size
    float accumulated_loss = 0.0f;
    int accumulated_count = 0;
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      // -1 表示该位置不参与监督；0 是合法 token id，不能再用 0 代表 ignore
      vector<int> target_ids(seq_len, -1);
      vector<int> step_input_ids(seq_len, -1);

      if (is_vlm) {
        if (vlm_dataset.empty())
          continue;
        int idx = rand() % (int)vlm_dataset.size();
        string img_p =
            (fs::path(corpus_path) / ("train_" + to_string(idx) + ".jpg"))
                .string();
        const int img_tokens = 196;
        // 保底：VLM 分支可能不会填满整个 seq_len，必须清空未写入区域
        std::memset(input_tensor.data, 0,
                    input_tensor.rows * input_tensor.cols * sizeof(float));
        process_image_to_input(img_p, vision_proj, input_tensor, d_model);
        string ans = vlm_dataset[idx].value("answer", "");
        auto tokens = bpe_model.encode(ans, false);
        for (size_t i = 0; i + 1 < tokens.size() &&
                           (i + img_tokens < (size_t)seq_len - 1);
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
        if (all_text_tokens.size() <= (size_t)seq_len + 1)
          continue;
        int start = rand() % (int)(all_text_tokens.size() - seq_len - 1);
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

      accumulated_loss += total_loss;
      accumulated_count += count;
    }
    
    // 平均梯度
    if (batch_size > 1) {
      auto scale_grads = [&](Tensor *t) {
        for (int i = 0; i < t->rows * t->cols; i++) {
          t->grad[i] /= batch_size;
        }
      };
      scale_grads(projection.W);
      scale_grads(&embed.weights);
      if (is_vlm)
        scale_grads(vision_proj.W);
      for (auto b : blocks) {
        scale_grads(b->ffn1.W);
        scale_grads(b->ffn2.W);
        scale_grads(b->attn.W_q.W);
        scale_grads(b->attn.W_k.W);
        scale_grads(b->attn.W_v.W);
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
    strict_clip(projection.W);
    strict_clip(&embed.weights);
    if (is_vlm)
      strict_clip(vision_proj.W);
    for (auto b : blocks) {
      strict_clip(b->ffn1.W);
      strict_clip(b->ffn2.W);
      strict_clip(b->attn.W_q.W);
      strict_clip(b->attn.W_k.W);
      strict_clip(b->attn.W_v.W);
    }

    // 前 5% 线性 warmup，后续由自适应调度主导（避免后期 lr 过小）
    int warmup_steps = std::max(1, epochs / 20);
    float scheduled_lr = lr;
    if (epoch < warmup_steps) {
      scheduled_lr = lr * (float)(epoch + 1) / (float)warmup_steps;
    }
    float current_lr = scheduled_lr * lr_scale;

    projection.update(current_lr);
    for (auto b : blocks)
      b->update(current_lr);
    if (is_vlm)
      vision_proj.update(current_lr);
    embed.update(current_lr);

    // 使用累积的 loss 计算平均 loss
    float avg_loss = (accumulated_count > 0 ? accumulated_loss / accumulated_count : 0.0f);
    int count = accumulated_count;
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

    if (epoch % 1 == 0) {
      printf("Epoch %d | Loss: %.4f | EMA: %.4f | LR: %.6f | Scale: %.3f\n",
             epoch, avg_loss, (loss_ema > 0 ? loss_ema : avg_loss), current_lr,
             lr_scale);
    }

    if (save_point > 0 && (epoch + 1) % save_point == 0) {
      ofstream out_f(weight_path, ios::binary);
      if (out_f.is_open()) {
        embed.weights.save(out_f);
        vision_proj.save(out_f);
        for (auto b : blocks)
          b->save(out_f);
        projection.save(out_f);
        out_f.close();
        cout << "[System] Checkpoint saved (epoch " << (epoch + 1) << ")"
             << endl;
      }
    }
  }

  {
    ofstream out_f(weight_path, ios::binary);
    if (out_f.is_open()) {
      embed.weights.save(out_f);
      vision_proj.save(out_f);
      for (auto b : blocks)
        b->save(out_f);
      projection.save(out_f);
      out_f.close();
    }
  }
  cout << "[System] Model saved" << endl;
  for (auto b : blocks)
    delete b;
  return 0;
}
