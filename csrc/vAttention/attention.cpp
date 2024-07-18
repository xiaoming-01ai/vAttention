#include "attention.h"
#include "kernel.h"
#include "kvcache/index/metric_type.h"
#include "utils/timer.h"
#include "utils/util.h"


VATTN_NAMESPACE_BEGIN

Attention::Attention(const std::string &desc) 
    : cache_desc_(desc)
{ }

void Attention::cache_fp32(const void *k,
                           const void *v,
                           int seqs_len,
                           int head_cnt,
                           int head_dim,
                           int max_tokens,
                           cudaStream_t stream)
{
    if (kvcache == nullptr) {
        kvcache = std::make_shared<KVCache>(
            seqs_len, head_cnt, head_dim, max_tokens, cache_desc_);
        head_dim_ = head_dim;
        head_cnt_ = head_cnt; 
    }
    VATTN_THROW_IF_NOT_MSG(kvcache != nullptr, "Invalid KVCache");
    kvcache->cache_fp32(seqs_len, k, v, stream);
}

void Attention::forward_fp32(const float *q,
                             int q_head_cnt,
                             int q_head_dim,
                             int topk,
                             float scale,
                             float *output,
                             cudaStream_t stream)
{
    auto metric = get_metric(MetricType::METRIC_IP, VectorType::VECTOR_FP32);
    VATTN_ASSERT(metric != nullptr);
    VATTN_THROW_IF_NOT_MSG(q != nullptr, "Invalid query");
    VATTN_THROW_IF_NOT_MSG(output != nullptr, "Invalid output");
    VATTN_THROW_IF_NOT_FMT(q_head_dim == head_dim_,
                           "Invalid query head_dim(%d) != KV head_dim(%d)",
                           q_head_dim, head_dim_);

    // may not enouth seqs in kv cache
    topk = std::min<int>(topk, kvcache->seqs_len);

    size_t bytes = sizeof(float) * head_dim_;
    size_t kv_size = topk + kvcache->padding_cnt;

    std::vector<int> labels(topk * q_head_cnt);
    kvcache->search_fp32(q, q_head_cnt, topk, labels.data());

    std::vector<float> scores(q_head_dim * kv_size);
    for (int i = 0; i < q_head_cnt; ++i) {
        // softmax(q * k)
        int *labels_ptr = labels.data() + i * topk;
        float *scores_ptr = scores.data() + i * kv_size;
        const float *q_ptr = (const float *)q + i * q_head_dim;

        int group = i / head_cnt_;

        float max_dot = std::numeric_limits<float>::min();
        for (int j = 0; j < topk; ++j) {
            float dot = scale * (1 - metric(q_ptr, kvcache->get_prefill_key(labels_ptr[j], group), bytes));
            max_dot = std::max<float>(max_dot, dot);
            scores_ptr[j] = dot;
        }
        for (int j = 0; j < kvcache->padding_cnt; ++j) {
            float dot = scale * (1 - metric(q_ptr, kvcache->get_padding_key(j, group), bytes));
            max_dot = std::max<float>(max_dot, dot);
            scores_ptr[j + topk] = dot;
        }

        float exp_sum = 0;
        for (int j = 0; j < kv_size; ++j) {
            scores_ptr[j] = std::exp(scores_ptr[j] - max_dot);
            exp_sum += scores_ptr[j];
        }
        float inv_sum = 1.0f / exp_sum;
        for (int j = 0; j < kv_size; ++j) {
            scores_ptr[j] *= inv_sum;
        }
        // print_value<float>("softmax ", scores_ptr, kv_size, 32);

        // matmal V (prefill topk)
        fprintf(stderr, "process q head %d group %d topk %d padding_cnt %d exp_sum %f max_doc %f\n", 
            i, group, topk, kvcache->padding_cnt, exp_sum, max_dot);
        
        float *output_ptr = output + i * head_dim_;
        for (int j = 0; j < topk; ++j) {
            auto v_ptr = kvcache->get_prefill_value(labels_ptr[j], group);
            compute_v_fp32(output_ptr, (const float *)v_ptr, scores_ptr[j], head_dim_);
        }
        // matmul V (padding)
        for (int j = 0; j < kvcache->padding_cnt; ++j) {
            auto v_ptr = kvcache->get_padding_value(j, group);
            compute_v_fp32(output_ptr, (const float *)v_ptr, scores_ptr[j + topk], head_dim_);
        }
        break;
    }
}
    
void Attention::release()
{
    if (kvcache != nullptr) {
        kvcache->release();
        kvcache = nullptr;
    }
}

VATTN_NAMESPACE_END