#include "BPE.h"
#include <iostream>
#include <algorithm>
#include <cstdint>

namespace bpe {
BPETrainer::BPETrainer() { build_initial_vocab(); }
BPETrainer::BPETrainer(const BPEConfig& config) : config_(config) { build_initial_vocab(); }

void BPETrainer::build_initial_vocab() {
    vocab_.clear(); id_to_vocab_.clear();
    vocab_[config_.unk_token] = UNK_TOKEN_ID; id_to_vocab_[UNK_TOKEN_ID] = config_.unk_token;
    vocab_[config_.bos_token] = BOS_TOKEN_ID; id_to_vocab_[BOS_TOKEN_ID] = config_.bos_token;
    vocab_[config_.eos_token] = EOS_TOKEN_ID; id_to_vocab_[EOS_TOKEN_ID] = config_.eos_token;
    vocab_[config_.pad_token] = PAD_TOKEN_ID; id_to_vocab_[PAD_TOKEN_ID] = config_.pad_token;

    for (int i = 0; i < 256; ++i) {
        std::string s(1, (unsigned char)i);
        if (vocab_.find(s) == vocab_.end()) {
            TokenId new_id = (TokenId)vocab_.size();
            vocab_[s] = new_id;
            id_to_vocab_[new_id] = s;
        }
    }
}

bool BPETrainer::train_from_texts(const std::vector<std::string>& texts) {
    build_initial_vocab();
    std::unordered_map<std::string, size_t> word_freqs;
    for (const auto& text : texts) {
        auto it = std::sregex_iterator(text.begin(), text.end(), config_.pattern);
        for (; it != std::sregex_iterator(); ++it) word_freqs[it->str()]++;
    }

    std::map<std::vector<std::string>, size_t> splits;
    for (auto const& [word, freq] : word_freqs) {
        std::vector<std::string> chars;
        for (unsigned char c : word) chars.push_back(std::string(1, c));
        splits[chars] = freq;
    }

    while (vocab_.size() < config_.vocab_size) {
        std::map<std::pair<std::string, std::string>, size_t> pair_freqs;
        for (auto const& [chars, freq] : splits) {
            for (size_t i = 0; i < chars.size() - 1; ++i)
                pair_freqs[{chars[i], chars[i+1]}] += freq;
        }
        if (pair_freqs.empty()) break;
        auto best = std::max_element(pair_freqs.begin(), pair_freqs.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        if (best->second < config_.min_frequency) break;

        std::string f = best->first.first, s = best->first.second, m = f + s;
        TokenId nid = (TokenId)vocab_.size();
        vocab_[m] = nid; id_to_vocab_[nid] = m;
        merge_rules_.push_back({f, s, m, nid});

        std::map<std::vector<std::string>, size_t> next_splits;
        for (auto const& [chars, freq] : splits) {
            std::vector<std::string> next_chars;
            for (size_t i = 0; i < chars.size(); ++i) {
                if (i < chars.size()-1 && chars[i] == f && chars[i+1] == s) { next_chars.push_back(m); i++; }
                else next_chars.push_back(chars[i]);
            }
            next_splits[next_chars] = freq;
        }
        splits = std::move(next_splits);
    }
    return true;
}

bool BPETrainer::train_from_file(const std::string& path) {
    std::ifstream f(path); if(!f) return false;
    std::vector<std::string> lines; std::string line;
    while(std::getline(f, line)) lines.push_back(line);
    return train_from_texts(lines);
}

std::vector<TokenId> BPETrainer::encode(const std::string& text, bool add_special) const {
    std::vector<TokenId> ids;
    if (add_special) ids.push_back(BOS_TOKEN_ID);
    auto it = std::sregex_iterator(text.begin(), text.end(), config_.pattern);
    for (; it != std::sregex_iterator(); ++it) {
        for (const auto& t : apply_merges(it->str())) ids.push_back(vocab_.count(t) ? vocab_.at(t) : UNK_TOKEN_ID);
    }
    if (add_special) ids.push_back(EOS_TOKEN_ID);
    return ids;
}

std::vector<std::string> BPETrainer::apply_merges(const std::string& word) const {
    std::vector<std::string> tokens;
    for (unsigned char c : word) tokens.push_back(std::string(1, c));
    for (const auto& r : merge_rules_) {
        std::vector<std::string> next;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i < tokens.size()-1 && tokens[i] == r.first && tokens[i+1] == r.second) { next.push_back(r.merged); i++; }
            else next.push_back(tokens[i]);
        }
        tokens = std::move(next);
    }
    return tokens;
}

std::string BPETrainer::decode(const std::vector<TokenId>& ids, bool skip_special) const {
    std::string res = "";
    for (TokenId id : ids) {
        if (skip_special && id <= 3) continue;
        if (id_to_vocab_.count(id)) res += id_to_vocab_.at(id);
    }
    return res;
}

std::string BPETrainer::id_to_token(TokenId id) const {
    return id_to_vocab_.count(id) ? id_to_vocab_.at(id) : config_.unk_token;
}

bool BPETrainer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary); if(!out) return false;
    size_t sz = merge_rules_.size(); out.write((char*)&sz, sizeof(sz));
    for(const auto& r : merge_rules_) {
        size_t s1=r.first.size(), s2=r.second.size(), s3=r.merged.size();
        out.write((char*)&s1, sizeof(s1)); out.write(r.first.data(), s1);
        out.write((char*)&s2, sizeof(s2)); out.write(r.second.data(), s2);
        out.write((char*)&s3, sizeof(s3)); out.write(r.merged.data(), s3);
        out.write((char*)&r.token_id, sizeof(r.token_id));
    }
    return true;
}

bool BPETrainer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary); if(!in) return false;
    build_initial_vocab(); size_t sz; in.read((char*)&sz, sizeof(sz));
    for(size_t i=0; i<sz; ++i) {
        size_t s1, s2, s3; 
        in.read((char*)&s1, sizeof(s1)); std::string f(s1, ' '); in.read(&f[0], s1);
        in.read((char*)&s2, sizeof(s2)); std::string s(s2, ' '); in.read(&s[0], s2);
        in.read((char*)&s3, sizeof(s3)); std::string m(s3, ' '); in.read(&m[0], s3);
        TokenId id; in.read((char*)&id, sizeof(id));
        merge_rules_.push_back({f, s, m, id}); vocab_[m] = id; id_to_vocab_[id] = m;
    }
    return true;
}

void BPETrainer::clear() { vocab_.clear(); id_to_vocab_.clear(); merge_rules_.clear(); }

} // namespace bpe
