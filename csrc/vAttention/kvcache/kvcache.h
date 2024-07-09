#pragma once

#include "kvcache/index/cache_index.h"

VATTN_NAMESPACE_BEGIN

struct KVCache {
    int    head_dim;
    int    kv_head_num;
    const void **keys;
    // const void **vals;

    CacheIndexPtr index;

    KVCache(int d, const std::string &index_desc);

    bool bind_fp32(int heads, int seqs, const void **keys);
    bool bind_bf16(int heads, int seqs, const void **keys);

    int search(int heads, const float **q, int k, int **labels);
};
using KVCachePtr = std::shared_ptr<KVCache>;

VATTN_NAMESPACE_END