#include "kvcache/index/cache_index_fast_scan_pq.h"
#include "common/vattn_assert.h"
#include "utils/heap.h"
#include "utils/neighbor.h"
#include "utils/timer.h"
#include "utils/random.h"

VATTN_NAMESPACE_BEGIN

static void subsample_training_set(int seq, int dim, const float *x, int samples, float **x_out)
{
    std::vector<int> perm(seq);
    rand_perm(perm.data(), seq, 1234);
    *x_out = new float[samples * dim];
    for (int i = 0; i < samples; i++) {
        memcpy((*x_out) + i * dim, x + (size_t)perm[i] * dim, sizeof(float) * dim);
    }
}

CacheIndexFastScanPQ::CacheIndexFastScanPQ(int d, int M)
    : CacheIndex(d), M(M)
{ }

void CacheIndexFastScanPQ::bind_fp32(int heads, int seqs, const void **keys) 
{
    CacheIndex::bind_fp32(heads, seqs, keys);

    pq.resize(heads);
    Timer train_timer;
    for (int i = 0; i < heads; ++i) { 
        float *samples = nullptr;
        subsample_training_set(seqs * heads, vector_dim, (const float *)keys[0], 512, &samples); 
        std::shared_ptr<float[]> del_ptr(samples);
        pq[i].reset(new FastScanProductQuantizer(vector_dim, M));
        pq[i]->train(512, samples);
        break;
    }
    auto train_cost = train_timer.elapsed_milliseconds();

    Timer codec_timer;

    // assign up to 32, make sure enouth buffer to store codes
    int align_seqs = (seqs + 31) / 32 * 32;
    codes.resize(align_seqs * pq[0]->code_size * heads);

    for (int i = 0; i < heads; ++i) {
        pq[0]->compute_codes(
            seqs, (const float *)keys[i], codes.data() + i * align_seqs * pq[0]->code_size);
    }
    
    auto codec_cost = codec_timer.elapsed_milliseconds();

    fprintf(stderr, "bind %d head cost %fms, train cost %fms. codec cost %fms\n", 
        heads, codec_cost + train_cost, train_cost, codec_cost);
}

int CacheIndexFastScanPQ::search(int heads, const float **q, int k, int **labels) const
{
    VATTN_THROW_IF_NOT(k > 0);
    VATTN_THROW_IF_NOT(keys_seqs > k);
    VATTN_THROW_IF_NOT(labels != nullptr);

    int align_keys_seqs = (keys_seqs + 31 ) / 32 * 32;
    std::shared_ptr<int32_t[]> distances(
        (int *)aligned_alloc(64,  heads * align_keys_seqs * sizeof(int32_t)), free);

    int group = heads / keys_heads;
    for (int i = 0; i < heads; ++i) {

        // compute lookup table 
        std::vector<uint8_t> lut(pq[0]->M * pq[0]->ksub);
        pq[0]->compute_quantized_ip_lut(1, q[i], lut.data());
        
        int32_t *distances_ptr = distances.get() + i * align_keys_seqs;
        const uint8_t *codes_ptr = codes.data() + (i / group) * align_keys_seqs * pq[0]->code_size;

        // compute score 
        #pragma omp parallel for
        for (int j = 0; j < align_keys_seqs; j += 32) {
            pq[0]->compute_distance(codes_ptr + j * pq[0]->code_size, lut.data(), &distances_ptr[j]);
        }
    }
       
    #pragma omp parallel for
    for (int i = 0; i < heads; ++i) {
        const int32_t *distances_ptr = distances.get() + i * align_keys_seqs;
        Heap<Neighbor> topk_heap(k);
        for (int j = 0; j < keys_seqs; ++j) {
            topk_heap.push(Neighbor(j, distances_ptr[j]));
        }

        for (auto j = 0; j < topk_heap.size(); ++j) {
            labels[i][j] = topk_heap.data()[j].id;
        }
    }

    return k;
}
    
VATTN_NAMESPACE_END
