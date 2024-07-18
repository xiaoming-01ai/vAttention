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

    void forward_fp32(const float *q,
                      int head_cnt,
                      int head_dim,
                      int topk,
                      float scale,
                      float *output,
                      cudaStream_t stream);

    void release();

    int head_cnt_;
    int head_dim_;

    KVCachePtr kvcache{nullptr};

    std::string cache_desc_;
};
using AttentionPtr = std::shared_ptr<Attention>;

struct AttentionManager {

    static AttentionManager *get_instance()
    {
        static AttentionManager ins;
        return &ins;
    }

    AttentionPtr fetch_attention(uint64_t attn) 
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        auto ite = attns.find(attn);
        VATTN_THROW_IF_NOT_FMT(ite != attns.end(), "Fetch invalid attention(%lu)", attn);
        return ite->second;
    }
    
    AttentionPtr create_attention(uint64_t attn)
    {
        std::lock_guard<std::mutex> lock(lock_mutex);
        auto ite = attns.find(attn);
        if (ite == nullptr) {
            auto attn_ptr = std::make_shared<Attention>(cache_desc);
            auto ret = attns.insert(std::make_pair(attn, attn_ptr));
            VATTN_THROW_IF_NOT_MSG(ret.second, "Insert attention failed.");
            ite = ret.first;
        }
        return ite->second;
    }

    void release_attention(uint64_t attn)
    {
        AttentionPtr attn_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(lock_mutex);
            auto ite = attns.find(attn);
            if (ite != attns.end()) {
                attn_ptr = ite->second;
                attns.erase(ite);
            }
        }
        if (attn_ptr != nullptr) {
            attn_ptr->release();
        }
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