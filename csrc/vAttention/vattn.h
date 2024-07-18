#pragma once

#include <stdio.h>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <limits>
#include <string.h>
#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define VATTN_NAMESPACE_BEGIN namespace vAttention { 
#define VATTN_NAMESPACE_END } 
#define VATTN_NAMESPACE_USE using namespace vAttention;
    
#define DELETE_AND_SET_NULLPTR(ptr) \
{ \
    delete[] ptr; \
    ptr = nullptr; \
}

VATTN_NAMESPACE_BEGIN

// void attention_cache_fp32(const uint64_t *attn_ids,
//                           int batch_size,
//                           const void *key,
//                           const void *value,
//                           int *seqs_len,
//                           int head_size,
//                           int head_dim,
//                           int max_tokens,
//                           cudaStream_t stream);

// void attention_forward_fp32(const uint64_t *attn_ids,
//                             int batch_size,
//                             const float *query,
//                             int head_size,
//                             int head_dim,
//                             float *output,
//                             cudaStream_t stream);

// void attention_finish(const uint64_t *attn_ids, 
//                       int batch_size);

VATTN_NAMESPACE_END