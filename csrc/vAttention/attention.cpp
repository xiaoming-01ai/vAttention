#include "attention.h"
#include "kernel.h"
#include "kvcache/index/metric_type.h"


VATTN_NAMESPACE_BEGIN

Attention::Attention(const std::string &desc) : cache_desc_(desc)
{ }

void Attention::cache_fp32(const void *k,
                           const void *v,
                           int seqs_len,
                           int head_cnt,
                           int head_dim,
                           int max_tokens,
                           cudaStream_t stream)
{
    if (kvcache_index == nullptr) {
        kvcache_index = std::make_shared<KVCache>(head_dim_, cache_desc_);
        head_cnt_ = head_cnt;
        head_dim_ = head_dim;
    }
    VATTN_ASSERT(kvcache_index != nullptr);
    VATTN_THROW_IF_NOT_FMT(head_cnt_ == head_cnt, 
                           "Invalid KV head_cnt(%d) != %d", head_cnt, head_cnt_);
    VATTN_THROW_IF_NOT_FMT(head_dim_ == head_dim, 
                           "Invalid KV head_dim(%d) != %d", head_dim, head_dim_);
    
    if (k_prefill_ == nullptr) {
        // first time cache KV cache, seqs > 1

        // allocate prefill KV cache buffer
        size_t bytes = sizeof(float) * seqs_len * head_dim_ * head_cnt_;
        k_prefill_ = new char[bytes];
        v_prefill_ = new char[bytes];

        // copy KV cache from GPU to host 
        cudaMemcpyAsync(k_prefill_ , k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_prefill_, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention copy FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));

        // bind host KV cache to index 
        std::vector<const void *> keys(head_cnt_);
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = k_prefill_ + sizeof(float) * seqs_len * head_dim_ * i; 
        }
        kvcache_index->bind_fp32(head_cnt_, seqs_len, keys.data());

        // allocate padding KV cache buffer for KV cache(forward_decode) 
        bytes = sizeof(float) * max_tokens * head_dim_ * head_cnt_;
        k_padding_ = new char[bytes];
        v_padding_ = new char[bytes];
        padding_cnt_ = 0;
        max_tokens_ = max_tokens;
    } else {
        // padding the decode_forward KV cahce, seqs = 1
        VATTN_THROW_IF_NOT_FMT(seqs_len + padding_cnt_ < max_tokens,
                               "Exceeded the maximun token limit(%d)", max_tokens);

        size_t bytes_per = sizeof(float) * head_dim_ * head_cnt_;
        size_t bytes = bytes_per * seqs_len;
        size_t padding_offset = padding_cnt_ * bytes_per;
        cudaMemcpyAsync(k_padding_ + padding_offset, k, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_padding_ + padding_offset, v, bytes, cudaMemcpyDeviceToHost, stream);
        auto cuda_error = cudaStreamSynchronize(stream);
        VATTN_THROW_IF_NOT_FMT(cuda_error == cudaSuccess,
                               "Attention padding FP32 K/V to host failed. cudaError(%s::%s)",
                               cudaGetErrorName(cuda_error), cudaGetErrorString(cuda_error));
        padding_cnt_ += seqs_len;
    }
}

std::vector<float> Attention::forward_decode_fp32(const void *q, 
                                                  int q_head_cnt, 
                                                  int q_head_dim, 
                                                  int topk)
{
    auto metric = get_metric(MetricType::METRIC_IP, VectorType::VECTOR_FP32);
    VATTN_ASSERT(metric != nullptr);

    std::vector<float> output(head_dim_ * head_cnt_);

    size_t bytes = sizeof(float) * head_dim_;
    size_t kv_size = topk + padding_cnt_;
    auto labels = search_fp32(q, q_head_cnt, q_head_dim, topk);
    std::vector<float> scores(q_head_dim * kv_size);
    for (int i = 0; i < q_head_cnt; ++i) {
        // softmax(q * k)
        int   *labels_ptr = labels.data() + i * topk;
        float *scores_ptr = scores.data() + i * kv_size;
        const float *q_ptr = (const float *)q + i * q_head_dim;
        const float *k_prefill_ptr = (const float *)k_prefill_ + i * head_cnt_ * head_dim_;

        float exp_sum = 0;
        for (int j = 0; j < topk; ++j) {
            float exp = std::exp(metric(q_ptr, k_prefill_ptr + labels_ptr[j] * head_dim_, bytes));
            exp_sum += exp;
            scores_ptr[j] = exp;
        }

        const float *k_padding_ptr = (const float *)k_padding_ + i * max_tokens_;
        for (int j = 0; j < padding_cnt_; ++j) {
            float exp = std::exp(metric(q_ptr, k_padding_ptr + j * head_dim_, bytes));
            exp_sum += exp;
            scores_ptr[j + topk] = exp;
        }

        for (int j = 0; j < kv_size; ++j) {
            scores_ptr[j] /= exp_sum;
        }

        // matmal V (prefill topk)
        float *output_ptr = output.data() + i * head_dim_;
        const float *v_prefill_ptr = (const float *)v_prefill_ + i * head_cnt_ * head_dim_;
        for (int j = 0; j < topk; ++j) {
            const float *v_ptr = v_prefill_ptr + labels_ptr[j] * head_dim_;
            compute_v_fp32(output_ptr, v_ptr, scores_ptr[j], head_dim_);
        }
        // matmul V (padding)
        const float *v_padding_ptr = (const float *)v_padding_ + i * max_tokens_;
        for (int j = 0; j < padding_cnt_; ++j) {
            const float *v_ptr = v_padding_ptr + j * head_dim_;
            compute_v_fp32(output_ptr, v_ptr, scores_ptr[j + topk], head_dim_);
        }
    }
    return output;
}
    
std::vector<int> Attention::search_fp32(const void *q, int q_head_cnt, int q_head_dim, int topk)
{
    VATTN_THROW_IF_NOT_FMT(head_dim_ == q_head_dim,
                           "Invalid Q head_dim(%d) != KV head_dim(%d)",
                            q_head_dim, head_dim_);

    VATTN_THROW_IF_NOT_FMT((head_cnt_ % q_head_cnt) == 0,
                            "Invalid Q head_cnt(%d) %% KV head_cnt(%d) != 0",
                            q_head_cnt, head_cnt_);

    VATTN_THROW_IF_NOT_FMT(q != nullptr, "Invalid Q(%p)", q);

    std::vector<const float *> qs(q_head_cnt);
    for (size_t i = 0; i < qs.size(); ++i) {
        qs[i] = (const float *)q + q_head_dim * i;
    }

    std::vector<int> label(topk * q_head_cnt);
    std::vector<int *> labels(q_head_cnt);
    for (size_t i = 0; i < labels.size(); ++i) {
        labels[i] = label.data() + i * topk;
    }
    kvcache_index->search(q_head_cnt, qs.data(), topk, labels.data());
    return label;
}
    
void Attention::release()
{
    DELETE_AND_SET_NULLPTR(k_prefill_);
    DELETE_AND_SET_NULLPTR(v_prefill_);
    DELETE_AND_SET_NULLPTR(k_padding_);
    DELETE_AND_SET_NULLPTR(v_padding_);
}

VATTN_NAMESPACE_END