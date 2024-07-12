#pragma once

#include "vattn.h"
#include "kvcache/kvcache.h"
#include "common/vattn_assert.h"

#include <mutex>
#include <unordered_map>

VATTN_NAMESPACE_BEGIN

class Attention {
public:
    Attention(const std::string &cache_desc);
    ~Attention() = default;

    void cache_fp32(const void *k, 
                    const void *v, 
                    int seqs_len, 
                    int head_cnt,
                    int head_dim,
                    int max_tokens,
                    cudaStream_t stream);
    
    std::vector<float> forward_decode_fp32(const void *q, int head_cnt, int head_dim, int topk);
    
    void release();

private:
    std::vector<int> search_fp32(const void *q, int head_cnt, int head_dim, int topk);

private:
    int head_cnt_;
    int head_dim_;

    int max_tokens_;
    int padding_cnt_;

    char *k_prefill_{nullptr};
    char *v_prefill_{nullptr};
    char *k_padding_{nullptr};
    char *v_padding_{nullptr};
    KVCachePtr kvcache_index{nullptr};

    std::string cache_desc_;
};
using AttentionPtr = std::shared_ptr<Attention>;

struct AttentionManager {

    static AttentionManager *get_instance()
    {
        static AttentionManager ins;
        return &ins;
    }

    AttentionPtr create_attention(uint64_t seq_id)
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        auto ite = attns.find(seq_id);
        if (ite == nullptr) {
            auto attn = std::make_shared<Attention>(cache_desc);
            auto ret = attns.insert(std::make_pair(seq_id, attn));
            VATTN_THROW_IF_NOT_MSG(ret.second, "Insert attention failed.");
            ite = ret.first;
        }
        return ite->second;
    }

    void release_attention(uint64_t seq_id)
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        auto ite = attns.find(seq_id);
        VATTN_THROW_IF_NOT_FMT(ite != attns.end(), 
                               "Invalid attention instance(%lu)", seq_id);

        auto attn = ite->second;
        attn->release();
        attns.erase(ite);
    }

    void set_cache_desc(const std::string &desc)
    {
        cache_desc = desc;
    }

    std::mutex lock_mutex;
    std::unordered_map<uint64_t, AttentionPtr> attns;
    std::string cache_desc{"FSPQ32"};
};

VATTN_NAMESPACE_END