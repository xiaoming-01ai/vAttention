// #include "vattn.h"
// #include "attention.h"

// VATTN_NAMESPACE_BEGIN

// void cache_fp32(const uint64_t *attn_ids, 
//                 int batch_size,
//                 const float *k, 
//                 const float *v, 
//                 int *seqs_len,
//                 int head_cnt,
//                 int head_dim,
//                 int max_tokens,
//                 cudaStream_t stream)
// {
//     for (int i = 0; i < batch_size; ++i) {
//         auto attn = AttentionManager::get_instance()->create_attention(attn_ids[i]);
//         attn->cache_fp32(k, v, seqs_len[i], head_cnt, head_dim, max_tokens, stream);
//         k += (std::size_t)seqs_len[i] * head_dim * head_cnt;
//         v += (std::size_t)seqs_len[i] * head_dim * head_cnt;
//     }
// }

// void attention_forward_fp32(const uint64_t *attn_ids,
//                             int batch_size,
//                             const float *query,
//                             int head_size,
//                             int head_dim,
//                             int topk,
//                             float *output,
//                             cudaStream_t stream)
// {
//     for (int i = 0; i < batch_size; ++i) {
//         auto attn = AttentionManager::get_instance()->fetch_attention(attn_ids[i]);
//         attn->forward_fp32(query + i * head_size * head_dim,
//                            head_size, 
//                            head_dim, 
//                            topk, 
//                            output + head_dim * head_size, 
//                            stream);
//     }
// }

// void attention_finish(const uint64_t *attn_ids, int batch_size)
// {
//     for (int i = 0; i < batch_size; ++i) {
//         AttentionManager::get_instance()->release_attention(attn_ids[i]);
//     }
// }

// VATTN_NAMESPACE_END