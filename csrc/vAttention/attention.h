#pragma once

#include "vattn.h"
#include "kvcache/kvcache.h"
#include "common/vattn_assert.h"

#include <mutex>
#include <unordered_map>

VATTN_NAMESPACE_BEGIN

class Attention {
public:
    Attention(int head_size, 
              int head_dim,
              int max_tokens,
              int min_seqs,
              const std::string &desc)
        : head_size(head_size), 
          head_dim(head_dim),
          max_tokens(max_tokens),
          min_seqs(min_seqs)
    { 
        kvcache_index = std::make_shared<KVCache>(head_dim, desc);
    }

    virtual ~Attention() = default;

    void cache_fp32(const void *k, const void *v, int seqs, cudaStream_t stream);
    void cache_bf16(const void *k, const void *v, int seqs, cudaStream_t stream);
    
    std::vector<float> forward_decode_fp32(const void *q, 
                                           int q_head_size, 
                                           int q_head_dim, 
                                           int topk);
    
    void release();

private:
    std::vector<int> search_fp32(const void *q, int q_head_size, int head_dim, int topk);
    // template <int ELEMENT_SIZE>
    // void sync_data(const void *key, const void *value, int seqs);

private:
    int head_size;
    int head_dim;
    int max_tokens;
    int min_seqs;

    int padding_cnt;

    char *k_prompt{nullptr};
    char *v_prompt{nullptr};
    char *k_padding{nullptr};
    char *v_padding{nullptr};
    KVCachePtr kvcache_index{nullptr};
};
using AttentionPtr = std::shared_ptr<Attention>;

struct AttentionManager {

    AttentionManager *get_instance()
    {
        static AttentionManager ins;
        return &ins;
    }

    uint64_t create_attention(int head_size, int head_dim)
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        return 0;
    }

    void release_attention(uint64_t attn_id)
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        auto ite = attentions.find(attn_id);
        VATTN_THROW_IF_NOT_FMT(ite != attentions.end(), 
                               "Invalid attention instance(%lu)", attn_id);

        auto attn = ite->second;
        attn->release();
        attentions.erase(ite);
    }

    std::mutex lock_mutex;
    std::unordered_map<uint64_t, AttentionPtr> attentions;
};

VATTN_NAMESPACE_END