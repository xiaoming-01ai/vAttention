#include "vattn.h"
#include "attention.h"

VATTN_NAMESPACE_BEGIN

void cache_fp32(const uint64_t *req_ids, 
                const float *k, 
                const float *v, 
                int batch_size,
                int *seqs_len,
                int head_cnt,
                int head_dim,
                int max_tokens,
                cudaStream_t stream)
{
    for (int i = 0; i < batch_size; ++i) {
        auto attn = AttentionManager::get_instance()->create_attention(req_ids[i]);
        attn->cache_fp32(k, v, seqs_len[i], head_cnt, head_dim, max_tokens, stream);
        k += (size_t)seqs_len[i] * head_dim * head_cnt;
        v += (size_t)seqs_len[i] * head_dim * head_cnt;
    }
}

void attention_forward(const uint64_t *seq_ids)
{

}

void attention_finish(const uint64_t *req_ids, int batch_size)
{
    for (int i = 0; i < batch_size; ++i) {
        AttentionManager::get_instance()->release_attention(req_ids[i]);
    }
}

VATTN_NAMESPACE_END