#include "../BPE.h"
#include "../data.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <thread>

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("turtle-mlx-bpe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        bpe::BPETrainer initial;
        bpe::BPEConfig config;
        config.vocab_size = initial.vocab_size() + 1;
        bpe::BPETrainer a(config), b(config);
        require(a.train_from_texts({"ab ab ab"}) && b.train_from_texts({"ba ba ba"}), "Training failed");
        const std::string unseen = "abababa";
        const auto a_ids = a.encode(unseen), b_ids = b.encode(unseen);
        require(a_ids != b_ids, "Fixture needs different merge rules");
        require(a.encode(unseen) == a_ids && b.encode(unseen) == b_ids, "Cross-model thread cache contamination");
        require(a.decode(a_ids) == unseen && b.decode(b_ids) == unseen, "Byte decoding failed");
        std::atomic<bool> passed{true};
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) threads.emplace_back([&] {
            for (int j = 0; j < 100; ++j)
                if (a.encode(unseen) != a_ids || b.encode(unseen) != b_ids) passed = false;
        });
        for (auto &thread : threads) thread.join();
        require(passed, "Concurrent encoding changed token IDs");
        a.add_special_token("<THINK>");
        auto a_path = (root / "a.bpe").string(), b_path = (root / "b.bpe").string();
        require(a.save(a_path, "dataset-a") && b.save(b_path, "dataset-b"), "Save failed");
        const auto fingerprint = a.fingerprint();
        const auto special = a.encode("<THINK>");
        require(special.size() == 1, "Added special token is not recognized");
        require(a.load(a_path, "dataset-a") && a.load(a_path, "dataset-a"), "Repeated load failed");
        require(a.fingerprint() == fingerprint && a.encode("<THINK>") == special && a.encode(unseen) == a_ids,
                "Reload changed vocabulary, special tokens or encoding");
        require(!a.load(b_path, "dataset-a") && a.fingerprint() == fingerprint, "Dataset mismatch changed the model");
        require(a.load(b_path) && a.encode(unseen) == b_ids, "Same-size load retained stale merge ranks");
        require(a.train_from_texts({"ab ab ab"}) && a.encode(unseen) == a_ids, "Retraining retained old rules");
        std::string bytes;
        for (int i = 0; i < 256; ++i) bytes.push_back(static_cast<char>(i));
        require(a.decode(a.encode(bytes)) == bytes, "Tokenizer lost binary bytes");
        const std::string code = "int main() {\r\n    return 世界;\n}\n";
        require(a.decode(a.encode(code, true), true) == code, "Code/Unicode round trip failed");
        const auto stable = a.fingerprint();
        std::ifstream saved(a_path, std::ios::binary);
        const std::string data((std::istreambuf_iterator<char>(saved)), {});
        const auto broken = (root / "broken.bpe").string();
        for (size_t cut : {size_t(0), size_t(1), size_t(13), data.size() - 1}) {
            std::ofstream out(broken, std::ios::binary); out.write(data.data(), cut); out.close();
            require(!a.load(broken) && a.fingerprint() == stable, "Truncated file changed tokenizer");
        }
        bool rejected = false;
        try { a.add_special_token("new", bpe::BASE_BYTE_OFFSET); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Conflicting forced ID was accepted");

        // Write a native-size OM_BPE_V1 fixture matching the uploaded format.
        const auto legacy = (root / "legacy.bpe").string();
        {
            std::ofstream out(legacy, std::ios::binary);
            uint32_t length = 9; uint64_t id_length = 0; size_t count = 1;
            out.write(reinterpret_cast<char*>(&length), 4); out << "OM_BPE_V1";
            out.write(reinterpret_cast<char*>(&id_length), 8); out.write(reinterpret_cast<char*>(&count), sizeof(count));
            for (std::string token : {"a", "b", "ab"}) {
                size_t size = token.size(); out.write(reinterpret_cast<char*>(&size), sizeof(size)); out << token;
            }
            auto id = static_cast<bpe::TokenId>(initial.vocab_size());
            out.write(reinterpret_cast<char*>(&id), sizeof(id));
        }
        require(a.load(legacy) && a.encode("ab") == std::vector<bpe::TokenId>{static_cast<bpe::TokenId>(initial.vocab_size())}, "Valid V1 file failed to load");
        require(a.encode("int").size() == 3, "Legacy lexer semantics changed");
        require(a.save(a_path) && b.load(a_path) && b.encode("int") == a.encode("int"), "V1 lexer was lost during V2 upgrade");

        const auto text_file = root / "short.txt";
        { std::ofstream out(text_file, std::ios::binary); out << "hello\n"; }
        auto corpus = read_training_corpus(text_file.string());
        require(corpus.texts == std::vector<std::string>{"hello\n"}, "Short file was dropped or modified");
        { std::ofstream out(text_file, std::ios::binary); out << std::string(2300, 'x'); }
        corpus = read_training_corpus(text_file.string());
        require(corpus.texts.size() == 2 && corpus.texts[0].size() == 2000 && corpus.texts[1].size() == 700,
                "Chunk overlap lost data");
        require(corpus.identity == read_training_corpus(text_file.string()).identity, "Dataset identity is not stable");
        fs::remove_all(root);
        std::cout << "MLX tokenizer/data tests passed\n";
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
