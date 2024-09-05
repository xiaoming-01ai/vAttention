#pragma once

#include "kvcache/index/cache_index.h"
#include "kvcache/index/vector_type.h"

VATTN_NAMESPACE_BEGIN

struct KVCache {
    KVCache(int seqs_len, int head_cnt, int head_dim, int max_tokens, const std::string &desc);

    void cache_fp32(int seqs, const void *k, const void *v, cudaStream_t stream);
    void cache_bf16(int seqs, const void *k, const void *v, cudaStream_t stream);

    int search_fp32(const void *q, int q_head_size, int k, int *labels);
    int search_bf16(const void *q, int q_head_size, int k, int *labels)
    {
        VATTN_ASSERT_MSG(false, "BF16 search not support now");
    }
    
    const void *get_prefill_key(int seqs, int head = 0) const;
    const void *get_prefill_value(int seqs, int head = 0) const;
    
    const void *get_padding_key(int token, int head = 0) const;
    const void *get_padding_value(int token, int head = 0) const;

    void release();

    VectorType vector_type;
    
    int seqs_len;
    int head_cnt;
    int head_dim;

    int max_tokens;
    int padding_cnt;

    // [seqs_len, head_size, head_dim]
    char *k_prefill{nullptr};
    // [seqs_len, head_size, head_dim]
    char *v_prefill{nullptr};

    // [max_tokens, head_size, head_dim] 
    char *k_padding{nullptr};
    // [max_tokens, head_size, head_dim] 
    char *v_padding{nullptr};

    CacheIndexPtr index;
};
using KVCachePtr = std::shared_ptr<KVCache>;
    
inline const void *KVCache::get_prefill_key(int seqs, int head) const
{
    size_t ele_size = vector_type == VectorType::VECTOR_FP32 ? 4 : 2;
    // fprintf(stderr, "get prefill key seqs %d head %d %ld\n", seqs, head, ele_size);
    return k_prefill + ele_size * (seqs * head_dim * head_cnt + head * head_dim);
}

inline const void *KVCache::get_prefill_value(int seqs, int head) const
{
    size_t ele_size = vector_type == VectorType::VECTOR_FP32 ? 4 : 2;
    // fprintf(stderr, "get prefill value seqs %d head %d %ld\n", seqs, head, ele_size);
    return v_prefill + ele_size * (seqs * head_dim * head_cnt + head * head_dim);
}

inline const void *KVCache::get_padding_key(int token, int head) const
{
    size_t ele_size = vector_type == VectorType::VECTOR_FP32 ? 4 : 2;
    // fprintf(stderr, "get padding key token %d head %d %ld\n", token, head, ele_size);
    return k_padding + ele_size * (token * head_dim * head_cnt + head * head_dim);
}

inline const void *KVCache::get_padding_value(int token, int head) const
{
    size_t ele_size = vector_type == VectorType::VECTOR_FP32 ? 4 : 2;
    // fprintf(stderr, "get padding value token %d head %d %ld\n", token, head, ele_size);
    return v_padding + ele_size * (token * head_dim * head_cnt + head * head_dim);
}

VATTN_NAMESPACE_END