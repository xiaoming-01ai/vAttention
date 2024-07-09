#include "kvcache/index/cache_index_flat.h"
#include "common/vattn_assert.h"
#include "utils/heap.h"
#include "utils/neighbor.h"

VATTN_NAMESPACE_BEGIN

void CacheIndexFlat::bind_fp32(int heads, int seqs, const void **keys) 
{
    CacheIndex::bind_fp32(heads, seqs, keys);
    metric = get_metric(METRIC_IP, VECTOR_FP32);
}
    
int CacheIndexFlat::search(int heads, const float **q, int k, int **labels) const
{
    if (heads <= 0) {
        return 0;
    }

    VATTN_THROW_IF_NOT(k > 0);
    VATTN_THROW_IF_NOT(labels != nullptr);

    // not enough keys to search. return all keys
    if (k >= keys_seqs) {
        for (int i = 0; i < keys_seqs; ++i) {
            labels[0][i] = i;
        }
        for (int i = 1; i < heads; ++i) {
            memcpy(labels[i], labels[0], sizeof(int) * keys_seqs);
        }
        return keys_seqs;
    }

    std::vector<float> scores((size_t)heads * keys_seqs);

    int group = heads / keys_heads;
    for (int i = 0; i < heads; ++i) {
        const char *vector_ptr = (const char *)vector_data[i / group];
        float *scores_ptr = scores.data() + (size_t)i * keys_seqs;
        #pragma omp parallel for num_threads(heads / 2)
        for (int j = 0; j < keys_seqs; ++j) {
            scores_ptr[j] = metric(q[i], vector_ptr + j * vector_bytes, vector_bytes);
        }
    }
        
    // #pragma omp parallel for num_threads(heads / 2)
    for (int i = 0; i < heads; ++i) {
        const float *scores_ptr = scores.data() + i * keys_seqs;
        Heap<Neighbor> topk_heap(k);
        for (int j = 0; j < keys_seqs; ++j) {
            topk_heap.push(Neighbor(j, scores_ptr[j]));
        }

        for (auto j = 0; j < topk_heap.size(); ++j) {
            labels[i][j] = topk_heap.data()[j].id;
        }
    }

    return k;
}

VATTN_NAMESPACE_END