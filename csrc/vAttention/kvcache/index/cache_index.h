#pragma once

#include "vattn.h"

#include <vector>

VATTN_NAMESPACE_BEGIN

struct CacheIndex {
    int vector_dim;         ///< vector dimension
    int vector_bytes;       ///< vector bytes 
    int keys_seqs;          ///< total seq of indexed vectors per head
    int keys_heads;         ///< total head of indexed
    std::vector<const void *> vector_data; // < vector data, shape[head_num, seq, head_dim]

    explicit CacheIndex(int head_dim = 128);

    virtual ~CacheIndex() = default;

    // void bind_fp32(int heads, int seqs, const void *keys);
    // void bind_bf16(int heads, int seqs, const void *keys);
    
    int search(int head, const float *q, int k, int *labels) const;
    
    virtual void bind_fp32(int heads, int seqs, const void **keys);
    virtual void bind_bf16(int heads, int seqs, const void **keys);
    
    virtual int search(int head, const float **q, int k, int **labels) const = 0;
};
using CacheIndexPtr = std::shared_ptr<CacheIndex>;

VATTN_NAMESPACE_END