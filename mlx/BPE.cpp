#include "BPE.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace bpe {
namespace {
constexpr const char* magic_v1 = "OM_BPE_V1";
constexpr const char* magic_v2 = "OM_BPE_V2";
void write_uint(std::ostream& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
void write_string(std::ostream& out, const std::string& value) {
    write_uint(out, value.size(), 8);
    out.write(value.data(), value.size());
}
}
BPETrainer::BPETrainer() { build_initial_vocab(); }
BPETrainer::BPETrainer(const BPEConfig& config) : config_(config) { build_initial_vocab(); }
void BPETrainer::build_initial_vocab() {
    clear();
    auto add = [&](const std::string& value) {
        if (!vocab_.count(value)) {
            TokenId id = static_cast<TokenId>(vocab_.size());
            vocab_[value] = id;
            id_to_vocab_[id] = value;
        }
    };
    for (const auto& value : {config_.unk_token, config_.bos_token, config_.eos_token,
                             config_.pad_token, std::string("<file_b>"), std::string("<file_e>")}) add(value);
    if (vocab_.size() != BASE_BYTE_OFFSET) throw std::invalid_argument("Special tokens must be distinct");
    for (int i = 0; i < 256; ++i) add(std::string(1, static_cast<char>(i)));
    if (vocab_.size() != BASE_BYTE_OFFSET + 256) throw std::invalid_argument("Special tokens cannot be single bytes");
    for (const auto& keyword : {"\n", "    ", "int", "return", "if", "else", "for", "while",
         "class", "struct", "public", "private", "protected", "const", "static", "void", "bool",
         "true", "false", "nullptr", "template", "typename", "using", "namespace", "#include",
         "import", "def", "self"}) add(keyword);
    rebuild_lookup();
}
void BPETrainer::rebuild_lookup() {
    merge_ranks_.clear();
    for (size_t i = 0; i < merge_rules_.size(); ++i)
        merge_ranks_[{merge_rules_[i].first, merge_rules_[i].second}] = i;
    special_tokens_ = {config_.unk_token, config_.bos_token, config_.eos_token, config_.pad_token, "<file_b>", "<file_e>"};
    special_tokens_.insert(special_tokens_.end(), additional_tokens_.begin(), additional_tokens_.end());
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size() || (a.size() == b.size() && a < b);
    });
    cache_key_ = std::make_shared<const char>(0);
}
void BPETrainer::add_special_token(const std::string& token, TokenId forced_id) {
    if (token.empty()) throw std::invalid_argument("Special token cannot be empty");
    auto existing = vocab_.find(token);
    if (existing != vocab_.end()) {
        if (forced_id >= 0 && forced_id != existing->second) throw std::invalid_argument("Special token ID conflicts");
        return;
    }
    const auto id = static_cast<TokenId>(vocab_.size());
    if (forced_id >= 0 && forced_id != id) throw std::invalid_argument("Forced token ID must be the next contiguous ID");
    if (vocab_.size() >= static_cast<size_t>(std::numeric_limits<TokenId>::max())) throw std::length_error("Vocabulary too large");
    vocab_[token] = id;
    id_to_vocab_[id] = token;
    additional_tokens_.push_back(token);
    cached_tokens_.clear();
    rebuild_lookup();
}
std::vector<std::string> BPETrainer::split(const std::string& text) const {
    std::vector<std::string> words;
    if (legacy_lexer_) {
        std::string word;
        for (unsigned char byte : text) {
            if (std::isspace(byte) || std::ispunct(byte)) {
                if (!word.empty()) { words.push_back(word); word.clear(); }
                words.emplace_back(1, static_cast<char>(byte));
            } else {
                word.push_back(static_cast<char>(byte));
                if (word.size() == 100) { words.push_back(word); word.clear(); }
            }
        }
        if (!word.empty()) words.push_back(word);
        return words;
    }
    size_t i = 0;
    auto special_at = [&](size_t pos) -> std::string {
        for (const auto& token : special_tokens_)
            if (text.compare(pos, token.size(), token) == 0) return token;
        return {};
    };
    while (i < text.size()) {
        auto special = special_at(i);
        if (!special.empty()) { words.push_back(special); i += special.size(); continue; }
        const size_t start = i++;
        const unsigned char first = static_cast<unsigned char>(text[start]);
        if (first == '\r' && i < text.size() && text[i] == '\n') ++i;
        else if (first == ' ' || first == '\t') {
            while (i < text.size() && i - start < 100 && (text[i] == ' ' || text[i] == '\t')) ++i;
        } else if (std::isalnum(first) || first == '_' || first >= 128) {
            while (i < text.size() && i - start < 100) {
                const unsigned char c = static_cast<unsigned char>(text[i]);
                if ((!std::isalnum(c) && c != '_' && c < 128) || !special_at(i).empty()) break;
                ++i;
            }
        }
        words.push_back(text.substr(start, i - start));
    }
    return words;
}
bool BPETrainer::train_from_texts(const std::vector<std::string>& texts) {
    auto additional = additional_tokens_;
    build_initial_vocab();
    for (const auto& token : additional) add_special_token(token);
    if (config_.vocab_size < vocab_.size()) return false;
    std::map<std::string, size_t> frequencies;
    for (const auto& text : texts) for (const auto& word : split(text)) ++frequencies[word];
    struct Word { std::string text; std::vector<TokenId> ids; size_t frequency; };
    std::vector<Word> words;
    for (const auto& [text, frequency] : frequencies) {
        Word word{text, {}, frequency};
        auto existing = vocab_.find(text);
        if (existing != vocab_.end()) word.ids.push_back(existing->second);
        else for (unsigned char c : text) word.ids.push_back(BASE_BYTE_OFFSET + c);
        words.push_back(std::move(word));
    }
    while (vocab_.size() < config_.vocab_size) {
        std::map<std::pair<TokenId, TokenId>, size_t> counts;
        for (const auto& word : words)
            for (size_t i = 0; i + 1 < word.ids.size(); ++i) counts[{word.ids[i], word.ids[i + 1]}] += word.frequency;
        if (counts.empty()) break;
        auto best = std::max_element(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
        if (best->second < config_.min_frequency) break;
        const auto [a, b] = best->first;
        const auto first = id_to_vocab_.at(a), second = id_to_vocab_.at(b), merged = first + second;
        auto existing = vocab_.find(merged);
        const TokenId id = existing == vocab_.end() ? static_cast<TokenId>(vocab_.size()) : existing->second;
        vocab_[merged] = id;
        id_to_vocab_[id] = merged;
        merge_rules_.push_back({first, second, merged, id});
        for (auto& word : words) {
            size_t write = 0;
            for (size_t read = 0; read < word.ids.size(); ++read) {
                if (read + 1 < word.ids.size() && word.ids[read] == a && word.ids[read + 1] == b) { word.ids[write++] = id; ++read; }
                else word.ids[write++] = word.ids[read];
            }
            word.ids.resize(write);
        }
    }
    rebuild_lookup();
    for (const auto& word : words) fast_vocab_[word.text] = word.ids;
    return true;
}
bool BPETrainer::train_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string text((std::istreambuf_iterator<char>(in)), {});
    return train_from_texts({text});
}
std::vector<std::string> BPETrainer::apply_merges(const std::string& word) const {
    std::vector<std::string> tokens;
    for (unsigned char c : word) tokens.emplace_back(1, static_cast<char>(c));
    while (tokens.size() > 1) {
        size_t rank = merge_rules_.size(), index = 0;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto found = merge_ranks_.find({tokens[i], tokens[i + 1]});
            if (found != merge_ranks_.end() && found->second < rank) { rank = found->second; index = i; }
        }
        if (rank == merge_rules_.size()) break;
        tokens[index] = merge_rules_[rank].merged;
        tokens.erase(tokens.begin() + index + 1);
    }
    return tokens;
}
std::vector<TokenId> BPETrainer::encode_fast(const std::string& text, bool add_special) const {
    struct Cache { std::shared_ptr<const char> key; std::unordered_map<std::string, std::vector<TokenId>> words; };
    thread_local Cache cache;
    if (cache.key != cache_key_) { cache.key = cache_key_; cache.words.clear(); }
    std::vector<TokenId> ids;
    if (add_special) ids.push_back(BOS_TOKEN_ID);
    for (const auto& word : split(text)) {
        auto direct = vocab_.find(word);
        if (!legacy_lexer_ && direct != vocab_.end()) { ids.push_back(direct->second); continue; }
        auto trained = fast_vocab_.find(word);
        if (trained != fast_vocab_.end()) { ids.insert(ids.end(), trained->second.begin(), trained->second.end()); continue; }
        auto found = cache.words.find(word);
        if (found == cache.words.end()) {
            std::vector<TokenId> encoded;
            for (const auto& token : apply_merges(word)) encoded.push_back(vocab_.at(token));
            if (cache.words.size() >= 4096) cache.words.clear();
            found = cache.words.emplace(word, std::move(encoded)).first;
        }
        ids.insert(ids.end(), found->second.begin(), found->second.end());
    }
    if (add_special) ids.push_back(EOS_TOKEN_ID);
    return ids;
}
std::vector<TokenId> BPETrainer::encode(const std::string& text, bool add_special) const { return encode_fast(text, add_special); }
std::string BPETrainer::decode(const std::vector<TokenId>& ids, bool skip_special) const {
    std::string text;
    for (auto id : ids) {
        if (skip_special && id >= UNK_TOKEN_ID && id < BASE_BYTE_OFFSET) continue;
        auto found = id_to_vocab_.find(id);
        if (found != id_to_vocab_.end()) text += found->second;
    }
    return text;
}
std::string BPETrainer::id_to_token(TokenId id) const {
    auto found = id_to_vocab_.find(id);
    return found == id_to_vocab_.end() ? config_.unk_token : found->second;
}
std::string BPETrainer::fingerprint() const {
    uint64_t hash = 14695981039346656037ULL;
    auto integer = [&](uint64_t value) { for (int i = 0; i < 8; ++i) { hash ^= (value >> (8 * i)) & 255; hash *= 1099511628211ULL; } };
    auto token = [&](const std::string& value) { integer(value.size()); for (unsigned char byte : value) { hash ^= byte; hash *= 1099511628211ULL; } };
    integer(legacy_lexer_ ? 0 : 1);
    integer(vocab_.size());
    for (size_t id = 0; id < vocab_.size(); ++id) token(id_to_token(static_cast<TokenId>(id)));
    for (const auto& rule : merge_rules_) { token(rule.first); token(rule.second); integer(rule.token_id); }
    for (const auto& value : additional_tokens_) token(value);
    return std::to_string(hash);
}
void BPETrainer::set_cached_tokens(const std::vector<TokenId>& tokens) {
    for (auto id : tokens) if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) throw std::invalid_argument("Cached token ID is out of range");
    cached_tokens_ = tokens;
}
bool BPETrainer::save(const std::string& path, const std::string& dataset_id) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    write_uint(out, 9, 4); out.write(magic_v2, 9);
    write_string(out, dataset_id.empty() ? dataset_id_ : dataset_id);
    write_uint(out, legacy_lexer_ ? 0 : 1, 4);
    write_uint(out, vocab_.size(), 8);
    for (size_t id = 0; id < vocab_.size(); ++id) write_string(out, id_to_token(static_cast<TokenId>(id)));
    write_uint(out, additional_tokens_.size(), 8);
    for (const auto& token : additional_tokens_) write_uint(out, vocab_.at(token), 4);
    write_uint(out, merge_rules_.size(), 8);
    for (const auto& rule : merge_rules_) {
        write_uint(out, vocab_.at(rule.first), 4); write_uint(out, vocab_.at(rule.second), 4); write_uint(out, rule.token_id, 4);
    }
    out.close();
    return static_cast<bool>(out);
}
bool BPETrainer::load(const std::string& path, const std::string& expected_dataset_id) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    auto end = in.tellg();
    if (end < 13) return false;
    size_t remaining = static_cast<size_t>(end);
    in.seekg(0);
    auto read = [&](void* buffer, size_t size) {
        if (size > remaining || !in.read(static_cast<char*>(buffer), size)) return false;
        remaining -= size; return true;
    };
    auto uint = [&](uint64_t& value, int bytes) {
        if (static_cast<size_t>(bytes) > remaining) return false;
        value = 0;
        for (int i = 0; i < bytes; ++i) { const int byte = in.get(); if (byte < 0) return false; value |= uint64_t(byte) << (8 * i); }
        remaining -= bytes; return true;
    };
    auto string = [&](std::string& value, bool legacy, bool allow_empty = false) {
        uint64_t size = 0;
        if (legacy) { size_t native = 0; if (!read(&native, sizeof(native))) return false; size = native; }
        else if (!uint(size, 8)) return false;
        if ((!allow_empty && size == 0) || size > remaining) return false;
        value.resize(static_cast<size_t>(size));
        return read(value.data(), value.size());
    };
    uint64_t magic_size = 0;
    if (!uint(magic_size, 4) || magic_size != 9) return false;
    std::string magic(9, '\0');
    if (!read(magic.data(), 9) || (magic != magic_v1 && magic != magic_v2)) return false;
    const bool legacy = magic == magic_v1;
    BPETrainer candidate(config_);
    if (!string(candidate.dataset_id_, false, true) || candidate.dataset_id_.size() > (1 << 20) ||
        (!expected_dataset_id.empty() && expected_dataset_id != candidate.dataset_id_)) return false;
    size_t initial = candidate.vocab_.size();
    if (!legacy) {
        uint64_t lexer = 0;
        if (!uint(lexer, 4) || lexer > 1) return false;
        candidate.legacy_lexer_ = lexer == 0;
        uint64_t count = 0;
        if (!uint(count, 8) || count < initial || count > remaining / 9 || count >= uint64_t(std::numeric_limits<TokenId>::max())) return false;
        const auto initial_tokens = candidate.id_to_vocab_;
        candidate.vocab_.clear(); candidate.id_to_vocab_.clear();
        for (uint64_t id = 0; id < count; ++id) {
            std::string token;
            if (!string(token, false) || candidate.vocab_.count(token) ||
                (id < initial && token != initial_tokens.at(static_cast<TokenId>(id)))) return false;
            candidate.vocab_[token] = static_cast<TokenId>(id); candidate.id_to_vocab_[static_cast<TokenId>(id)] = token;
        }
        uint64_t extra = 0;
        if (!uint(extra, 8) || extra > remaining / 4 || extra > count - initial) return false;
        std::unordered_set<TokenId> added;
        for (uint64_t i = 0; i < extra; ++i) {
            uint64_t id = 0;
            if (!uint(id, 4) || id < initial || id >= count || !added.insert(static_cast<TokenId>(id)).second) return false;
            candidate.additional_tokens_.push_back(candidate.id_to_token(static_cast<TokenId>(id)));
        }
        std::vector<bool> available(count, false);
        for (size_t i = 0; i < initial; ++i) available[i] = true;
        for (auto id : added) available[id] = true;
        uint64_t rules = 0;
        if (!uint(rules, 8) || rules > remaining / 12) return false;
        std::set<std::pair<TokenId, TokenId>> pairs;
        for (uint64_t i = 0; i < rules; ++i) {
            uint64_t a = 0, b = 0, id = 0;
            if (!uint(a, 4) || !uint(b, 4) || !uint(id, 4) || a >= count || b >= count || id >= count ||
                !available[a] || !available[b] || !pairs.insert({static_cast<TokenId>(a), static_cast<TokenId>(b)}).second) return false;
            auto first = candidate.id_to_token(a), second = candidate.id_to_token(b), merged = candidate.id_to_token(id);
            if (merged != first + second) return false;
            candidate.merge_rules_.push_back({first, second, merged, static_cast<TokenId>(id)}); available[id] = true;
        }
        if (std::find(available.begin(), available.end(), false) != available.end()) return false;
    } else {
        candidate.legacy_lexer_ = true;
        size_t count = 0;
        if (!read(&count, sizeof(count)) || count > remaining / (3 * sizeof(size_t) + sizeof(TokenId) + 4)) return false;
        for (size_t i = 0; i < count; ++i) {
            std::string a, b, merged;
            TokenId id = 0;
            if (!string(a, true) || !string(b, true) || !string(merged, true) || !read(&id, sizeof(id)) ||
                merged != a + b || !candidate.vocab_.count(a) || !candidate.vocab_.count(b)) return false;
            auto existing = candidate.vocab_.find(merged);
            const TokenId expected = existing == candidate.vocab_.end() ? static_cast<TokenId>(candidate.vocab_.size()) : existing->second;
            if (id != expected) return false; // The local trainer sometimes reused IDs for different tokens.
            candidate.vocab_[merged] = id; candidate.id_to_vocab_[id] = merged;
            candidate.merge_rules_.push_back({a, b, merged, id});
        }
    }
    if (remaining != 0) return false;
    candidate.rebuild_lookup();
    *this = std::move(candidate);
    return true;
}
void BPETrainer::clear() {
    legacy_lexer_ = false;
    vocab_.clear(); id_to_vocab_.clear(); merge_rules_.clear(); merge_ranks_.clear(); fast_vocab_.clear();
    additional_tokens_.clear(); special_tokens_.clear(); dataset_id_.clear(); cached_tokens_.clear();
    cache_key_ = std::make_shared<const char>(0);
}
}
