#pragma once

#include "vattn.h"

#include <vector>

VATTN_NAMESPACE_BEGIN

struct CacheIndex {
    int vector_dim;   ///< vector dimension
    int vector_bytes; ///< vector bytes 
    int keys_seqs;    ///< total seq of indexed vectors per head
    int keys_heads;   ///< total head of indexed
    const void *keys; /// < vector data, shape[seqs_len, head_size, head_dim]

    explicit CacheIndex(int head_dim = 128);

    virtual ~CacheIndex() = default;

    // void bind_fp32(int heads, int seqs, const void *keys);
    // void bind_bf16(int heads, int seqs, const void *keys);
    
    int search(int head, const float *q, int k, int *labels) const;
    
    // shape [seqs, head_size, head_dim]
    virtual void bind_fp32(int seqs_len, int head_size, const void *keys);
    // shape [seqs, head_size, head_dim]
    virtual void bind_bf16(int seqs_len, int head_size, const void *keys);
    
    virtual int search(int head, const float **q, int k, int **labels) const = 0;
    
    const void *get_key(int seqs, int head = 0) const;
};
using CacheIndexPtr = std::shared_ptr<CacheIndex>;
    
inline const void *CacheIndex::get_key(int seqs, int head) const
{
    return (const char *)keys + seqs * keys_heads * vector_bytes + head * vector_bytes;
}

VATTN_NAMESPACE_END