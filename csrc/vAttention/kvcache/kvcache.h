#pragma once

#include "kvcache/index/cache_index.h"
#include "kvcache/index/vector_type.h"

VATTN_NAMESPACE_BEGIN

struct KVCache {
    KVCache(int d, const std::string &index_desc);

    void bind_fp32(int heads, int seqs, const void **keys);
    void bind_bf16(int heads, int seqs, const void **keys);

    int search(int heads, const float **q, int k, int **labels);

    VectorType vector_type;
    int    head_dim;
    int    seqs_len;
    const void **keys;
    const void **vals;

    CacheIndexPtr index;
};
using KVCachePtr = std::shared_ptr<KVCache>;

VATTN_NAMESPACE_END