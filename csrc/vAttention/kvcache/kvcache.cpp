#include "kvcache.h"
#include "common/vattn_assert.h"
#include "kvcache/index/cache_index_flat.h"
#include "kvcache/index/cache_index_fast_scan_pq.h"
#include "utils/timer.h"

#include <regex>

VATTN_NAMESPACE_BEGIN

CacheIndexPtr create_index(int head_dim, const std::string &index_desc)
{
    VATTN_THROW_IF_NOT_FMT(head_dim != 0 && head_dim <= 65536,
                           "The head_dim %d shoule be > 0 and < 65536",
                           head_dim);
    
    std::smatch sm;
    auto match = [&sm, index_desc](const std::string& pattern) {
        return std::regex_match(index_desc, sm, std::regex(pattern));
    };

    CacheIndexPtr index = nullptr;
    fprintf(stderr, "create index: %s\n", index_desc.c_str());
    if (match("FSPQ([0-9]+)")) {
        int M = std::stoi(sm[1].str());
        // index.reset(new CacheIndexFastScanPQ(head_dim, M));
    } else if (match("FLAT")) {
        index.reset(new CacheIndexFlat(head_dim));
    } else { 
        VATTN_ASSERT_FMT(false, 
                         "Index description[%s] not support", 
                         index_desc.c_str());
    }
    return index;
}

KVCache::KVCache(int seqs_len,
                 int head_cnt,
                 int head_dim,
                 int max_tokens,
                 const std::string &index_desc)
    : seqs_len(seqs_len),
      head_cnt(head_cnt),
      head_dim(head_dim),
      max_tokens(max_tokens),
      padding_cnt(0),
      k_prefill(nullptr),
      v_prefill(nullptr),
      k_padding(nullptr),
      v_padding(nullptr)
{
    index = create_index(head_dim, index_desc);
}

// shape: [seqs_len, head_size, head_dim]
void KVCache::cache_fp32(int seqs, const void *k, const void *v, cudaStream_t stream)
{
    Timer time;
    vector_type = VectorType::VECTOR_FP32;

    // first prefill call
    if (index->keys_seqs == 0) { 
        fprintf(stderr, "cache ...............\n");
        // allocate prefill KV cache buffer
        size_t bytes = sizeof(float) * seqs_len * head_dim * head_cnt;
        k_prefill = new (std::nothrow) char[bytes];
        VATTN_THROW_IF_NOT_FMT(k_prefill != nullptr,
                               "Allocate K cache failed. seqs_len(%d) head_dim(%d) head_cnt(%d)",
                               seqs_len, head_cnt, head_dim);

        v_prefill = new (std::nothrow) char[bytes];
        VATTN_THROW_IF_NOT_FMT(v_prefill != nullptr,
                               "Allocate V cache failed. seqs_len(%d) head_dim(%d) head_cnt(%d)",
                               seqs_len, head_cnt, head_dim);

        // allocate padding KV cache buffer for KV cache(forward_decode)
        bytes = sizeof(float) * max_tokens * head_dim * head_cnt;
        k_padding = new (std::nothrow) char[bytes];
        VATTN_THROW_IF_NOT_FMT(k_padding != nullptr,
                               "Allocate K padding cache failed. max_tokens(%d) head_dim(%d) head_cnt(%d)",
                               max_tokens, head_cnt, head_dim);

        v_padding = new char[bytes];
        VATTN_THROW_IF_NOT_FMT(v_padding != nullptr,
                               "Allocate V padding cache failed. max_tokens(%d) head_dim(%d) head_cnt(%d)",
                               max_tokens, head_cnt, head_dim);

        // copy KV cache from GPU to host
        bytes = sizeof(float) * head_cnt * head_dim * seqs_len;
        cudaMemcpyAsync(k_prefill, k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_prefill, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention copy FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));

        fprintf(stderr, "Copy %f MB bytes device to host cost %f ms\n",
                (float)bytes * 2 / 1024 / 1024, time.elapsed_milliseconds());

        // bind host KV cache to index
        index->bind_fp32(seqs_len, head_cnt, k_prefill);
    } else {
        fprintf(stderr, "padding...............\n");
        // padding the decode_forward KV cahce, seqs = 1
        VATTN_THROW_IF_NOT_FMT(seqs_len + padding_cnt < max_tokens,
                               "Exceeded the maximun token limit(%d)", 
                               max_tokens);

        std::size_t bytes_per = sizeof(float) * head_dim * head_cnt;
        std::size_t bytes = bytes_per * seqs_len;
        std::size_t padding_offset = padding_cnt * bytes_per;
        cudaMemcpyAsync(k_padding + padding_offset, k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_padding + padding_offset, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention padding FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));
        padding_cnt += seqs_len;
    }
}

int KVCache::search_fp32(const void *q, int q_head_size, int k, int *labels)
{
    return index->search(q_head_size, (const float *)q, k, labels);
}
    
void KVCache::release()
{
    DELETE_AND_SET_NULLPTR(k_prefill);
    DELETE_AND_SET_NULLPTR(v_prefill);
    DELETE_AND_SET_NULLPTR(k_padding);
    DELETE_AND_SET_NULLPTR(v_padding);
}
    
VATTN_NAMESPACE_END