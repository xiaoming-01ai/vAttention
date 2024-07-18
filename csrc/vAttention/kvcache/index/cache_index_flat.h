#pragma once

#include "kvcache/index/cache_index.h"
#include "kvcache/index/metric_type.h"

VATTN_NAMESPACE_BEGIN

struct CacheIndexFlat : public CacheIndex {
    Metric metric;

    explicit CacheIndexFlat(int d)
        : CacheIndex(d)
    { }

    void bind_fp32(int seqs_len, int head_size, const void *keys) override;
    
    int search(int heads, const float **q, int k, int **labels) const override;
};

VATTN_NAMESPACE_END