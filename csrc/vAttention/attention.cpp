#include "attention.h"
#include "kernel.h"
#include "kvcache/index/metric_type.h"


VATTN_NAMESPACE_BEGIN

Attention::Attention(int head_size, int head_dim, int max_tokens, const std::string &desc)
    : head_size(head_size),
      head_dim(head_dim),
      max_tokens(max_tokens)
{
    kvcache_index = std::make_shared<KVCache>(head_dim, desc);
}

void Attention::cache_fp32(const void *k, const void *v, int seqs, cudaStream_t stream)
{
    VATTN_ASSERT(kvcache_index != nullptr);
    if (k_prefill == nullptr) {
        // first time cache KV cache, seqs > 1

        // allocate prefill KV cache buffer
        size_t bytes = sizeof(float) * seqs * head_dim * head_size;
        k_prefill = new char[bytes];
        v_prefill = new char[bytes];

        // copy KV cache from GPU to host 
        cudaMemcpyAsync(k_prefill, k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_prefill, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention copy FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));

        // bind host KV cache to index 
        std::vector<const void *> keys(head_size);
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = k_prefill + sizeof(float) * seqs * head_dim * i; 
        }
        kvcache_index->bind_fp32(head_size, seqs, keys.data());

        // allocate padding KV cache buffer for KV cache(forward_decode) 
        bytes = sizeof(float) * max_tokens * head_dim * head_size;
        k_padding = new char[bytes];
        v_padding = new char[bytes];
        padding_cnt = 0;
    } else {
        // padding the decode_forward KV cahce, seqs = 1
        VATTN_THROW_IF_NOT_FMT(seqs + padding_cnt < max_tokens,
                               "Exceeded the maximun token limit(%d)", max_tokens);

        size_t bytes_per = sizeof(float) * head_dim * head_size;
        size_t bytes = bytes_per * seqs;
        size_t padding_offset = padding_cnt * bytes_per;
        cudaMemcpyAsync(k_padding + padding_offset, k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_padding + padding_offset, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention padding FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));
        padding_cnt += seqs;
    }
}

// void Attention::cache_bf16(const void *k, const void *v, int seqs, cudaStream_t stream)
// {
//     VATTN_ASSERT(kvcache_index != nullptr);
//     if (k_prefill == nullptr) {
//         size_t bytes = (size_t)2LL * seqs * head_dim * head_size;
//         k_prefill = new char[bytes];
//         v_prefill = new char[bytes];
        
//         cudaMemcpyAsync(k_prefill, k, bytes, cudaMemcpyDeviceToHost, stream);
//         cudaMemcpyAsync(v_prefill, v, bytes, cudaMemcpyDeviceToHost, stream);
//         auto cuda_error = cudaStreamSynchronize(stream);
//         VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
//                                "Attention copy BF16 K/V to host failed. cudaError(%s::%s)",
//                                cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));

//         bytes = 2 * max_tokens * head_dim * head_size;
//         k_padding = new char[bytes];
//         v_padding = new char[bytes];

//         std::vector<const void *> keys(head_size);
//         for (size_t i = 0; i < keys.size(); ++i) {
//             keys[i] = k_prefill + size_t(2L) * seqs * head_dim * i; 
//         }
//         kvcache_index->bind_bf16(head_size, seqs, keys.data());
//     } else {
//         VATTN_THROW_IF_NOT_FMT(seqs + padding_cnt < max_tokens,
//                                "Exceeded the maximun token limit(%d)", max_tokens);

//         size_t bytes = size_t(2L) * seqs * head_dim * head_size;
//         cudaMemcpyAsync(k_padding, k, bytes, cudaMemcpyDeviceToHost, stream);
//         cudaMemcpyAsync(v_padding, v, bytes, cudaMemcpyDeviceToHost, stream);
//         auto cuda_error = cudaStreamSynchronize(stream);
//         VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
//                                "Attention copy BF16 K/V to host failed. cudaError(%s::%s)",
//                                cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));
//     }
// }
    
std::vector<float> Attention::forward_decode_fp32(const void *q, 
                                                  int q_head_size, 
                                                  int q_head_dim, 
                                                  int topk)
{
    auto metric = get_metric(MetricType::METRIC_IP, VectorType::VECTOR_FP32);
    VATTN_ASSERT(metric != nullptr);

    std::vector<float> output(head_dim * head_size);

    size_t bytes = sizeof(float) * head_dim;
    size_t kv_size = topk + padding_cnt;
    auto labels = search_fp32(q, q_head_size, q_head_dim, topk);
    std::vector<float> scores(q_head_dim * kv_size);
    for (int i = 0; i < q_head_size; ++i) {
        // softmax(q * k)
        int   *labels_ptr = labels.data() + i * topk;
        float *scores_ptr = scores.data() + i * kv_size;
        const float *q_ptr = (const float *)q + i * q_head_dim;
        const float *k_prefill_ptr = (const float *)k_prefill + i * head_size * head_dim;

        float exp_sum = 0;
        for (int j = 0; j < topk; ++j) {
            float exp = std::exp(metric(q_ptr, k_prefill_ptr + labels_ptr[j] * head_dim, bytes));
            exp_sum += exp;
            scores_ptr[j] = exp;
        }

        const float *k_padding_ptr = (const float *)k_padding + i * max_tokens;
        for (int j = 0; j < padding_cnt; ++j) {
            float exp = std::exp(metric(q_ptr, k_padding_ptr + j * head_dim, bytes));
            exp_sum += exp;
            scores_ptr[j + topk] = exp;
        }

        for (int j = 0; j < kv_size; ++j) {
            scores_ptr[j] /= exp_sum;
        }

        // matmal V (prefill topk)
        float *output_ptr = output.data() + i * head_dim;
        const float *v_prefill_ptr = (const float *)v_prefill + i * head_size * head_dim;
        for (int j = 0; j < topk; ++j) {
            const float *v_ptr = v_prefill_ptr + labels_ptr[j] * head_dim;
            compute_v_fp32(output_ptr, v_ptr, scores_ptr[j], head_dim);
        }
        // matmul V (padding)
        const float *v_padding_ptr = (const float *)v_padding + i * max_tokens;
        for (int j = 0; j < padding_cnt; ++j) {
            const float *v_ptr = v_padding_ptr + j * head_dim;
            compute_v_fp32(output_ptr, v_ptr, scores_ptr[j + topk], head_dim);
        }
    }
    return output;
}
    
std::vector<int> Attention::search_fp32(const void *q, int q_head_size, int q_head_dim, int topk)
{
    VATTN_THROW_IF_NOT_FMT(head_dim == q_head_dim,
                           "Invalid Q head_dim(%d) != KV head_dim(%d)",
                            q_head_dim, head_dim);

    VATTN_THROW_IF_NOT_FMT((head_size % q_head_size) == 0,
                            "Invalid Q head_size(%d) %% KV head_size(%d) != 0",
                            q_head_size, head_size);

    VATTN_THROW_IF_NOT_FMT(q != nullptr, "Invalid Q(%p)", q);

    std::vector<const float *> qs(q_head_size);
    for (size_t i = 0; i < qs.size(); ++i) {
        qs[i] = (const float *)q + q_head_dim * i;
    }

    std::vector<int> label(topk * q_head_size);
    std::vector<int *> labels(q_head_size);
    for (size_t i = 0; i < labels.size(); ++i) {
        labels[i] = label.data() + i * topk;
    }
    kvcache_index->search(q_head_size, qs.data(), topk, labels.data());
    return label;
}
    
void Attention::release()
{
    DELETE_AND_SET_NULLPTR(k_prefill);
    DELETE_AND_SET_NULLPTR(v_prefill);
    DELETE_AND_SET_NULLPTR(k_padding);
    DELETE_AND_SET_NULLPTR(v_padding);
}

VATTN_NAMESPACE_END