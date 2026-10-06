#ifndef TURTLE_MLX_DATA_H
#define TURTLE_MLX_DATA_H
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TrainingCorpus {
  std::vector<std::string> texts;
  std::string identity;
};
inline TrainingCorpus read_training_corpus(const std::string &path,
                                          size_t chunk_size = 2000,
                                          size_t overlap = 400) {
  namespace fs = std::filesystem;
  if (chunk_size == 0 || overlap >= chunk_size)
    throw std::invalid_argument("Invalid corpus chunk configuration");
  std::vector<fs::path> files;
  if (fs::is_regular_file(path)) files.emplace_back(path);
  else if (fs::is_directory(path)) {
    for (const auto &entry : fs::directory_iterator(path))
      if (entry.is_regular_file() && entry.path().filename().string().front() != '.') files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::invalid_argument("No training files: " + path);
  uint64_t hash = 14695981039346656037ULL;
  auto digest = [&](const std::string &bytes) {
    for (unsigned char byte : bytes) { hash ^= byte; hash *= 1099511628211ULL; }
    hash ^= bytes.size(); hash *= 1099511628211ULL;
  };
  TrainingCorpus corpus;
  for (const auto &file : files) {
    digest(file.filename().string());
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read training file: " + file.string());
    if (files.size() == 1) {
      std::string prefix;
      while (in) {
        std::string bytes(chunk_size - prefix.size(), '\0');
        in.read(bytes.data(), bytes.size());
        bytes.resize(static_cast<size_t>(in.gcount()));
        if (bytes.empty()) break;
        digest(bytes);
        std::string chunk = prefix + bytes;
        corpus.texts.push_back(chunk);
        prefix = chunk.substr(chunk.size() > overlap ? chunk.size() - overlap : 0);
      }
      if (!in.eof() && in.fail()) throw std::runtime_error("Training file read failed");
    } else {
      std::string bytes((std::istreambuf_iterator<char>(in)), {});
      if (in.bad()) throw std::runtime_error("Training file read failed");
      digest(bytes);
      if (!bytes.empty()) corpus.texts.push_back(std::move(bytes));
    }
  }
  if (corpus.texts.empty()) throw std::invalid_argument("Training corpus is empty");
  corpus.identity = std::to_string(hash);
  return corpus;
}
#endif
