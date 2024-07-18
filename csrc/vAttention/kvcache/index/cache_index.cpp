#include "kvcache/index/cache_index.h"
#include "common/vattn_assert.h"

VATTN_NAMESPACE_BEGIN
    
CacheIndex::CacheIndex(int head_dim) 
    : vector_dim(head_dim),
      vector_bytes(0),
      keys_seqs(0),
      keys_heads(0), 
      keys(nullptr)
{ }
    
void CacheIndex::bind_fp32(int seqs_len, int head_size, const void *keys)
{
    vector_bytes = sizeof(float) * vector_dim;
    keys_seqs = seqs_len;
    keys_heads = head_size;
}

void CacheIndex::bind_bf16(int seqs_len, int head_size, const void *keys)
{
    VATTN_ASSERT_MSG(false, "BF16 not support now");
}

int CacheIndex::search(int heads, const float *q, int k, int *labels) const
{
    if (heads <= 0) {
        return 0;
    }

    std::vector<const float *> q_list(heads);
    std::vector<int *> labels_list(heads);

    for(int i = 0; i < heads; ++i) {
        q_list[i] = &q[i * vector_dim];
        labels_list[i] = &labels[i * k];
    }
    return search(heads, q_list.data(), k, labels_list.data());
}
    
VATTN_NAMESPACE_END