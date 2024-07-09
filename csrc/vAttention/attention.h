#pragma once

#include "vattn.h"
#include "common/vattn_assert.h"

#include <mutex>
#include <unordered_map>

VATTN_NAMESPACE_BEGIN

struct Attention {

    Attention(int head_size, int head_dim)
        : head_size(head_size), head_dim(head_dim)
    { }

    virtual ~Attention() = default;

    bool cache(const void *key, const void *value, int kv_cnt);

    bool forward_prefix(const void *query);
    bool forward_decode(const void *query);

    void release()
    { }

    int head_size;
    int head_dim;
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