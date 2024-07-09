#include "common/vattn_assert.h"
#include "kvcache/index/product_quantizer.h"
#include "kvcache/index/metric_type.h"
#include "kvcache/index/vector_type.h"

VATTN_NAMESPACE_BEGIN

ProductQuantizer::ProductQuantizer(size_t d, size_t M, size_t nbits)
    : Quantizer(d, 0), M(M), nbits(nbits) 
{
    VATTN_THROW_IF_NOT_FMT(
        d % M == 0,
        "The dimension of the vector %ld should be a multiple of the number of subquantizers %ld",
        d, M);

    dsub = d / M;
    ksub = 1 << nbits;
    code_size = (nbits * M + 7) / 8;
    centroids.resize(d * ksub);
    fprintf(stderr, "dsub=%ld ksub=%ld code=%ld M=%ld\n", dsub, ksub, code_size, M);
}

void ProductQuantizer::train(size_t n, const float* x) 
{
    std::unique_ptr<float[]> xslice(new float[n * dsub]);
    for (size_t m = 0; m < M; m++) {
        for (size_t j = 0; j < n; j++) {
            memcpy(xslice.get() + j * dsub, x + j * d + m * dsub, dsub * sizeof(float));
        }
        Clustering clus(dsub, ksub);
        clus.train(n, xslice.get());
        memcpy(get_centroids(m, 0), clus.centroids.data(), ksub * dsub * sizeof(clus.centroids[0]));
    }

    // for (int mi = 0; mi < M; ++mi) {
    //     for (int ksubi = 0; ksubi < ksub; ++ksubi) {
    //         fprintf(stderr, "M[%d][%d]=", mi, ksubi);
    //         for (int dsubi = 0; dsubi < dsub; ++dsubi) {
    //             fprintf(stderr, "%f ", get_centroids(mi, ksubi)[dsubi]);
    //         }
    //         fprintf(stderr, "\n");
    //     }
    // }
}

void ProductQuantizer::compute_code(const float* x, uint16_t* code) const 
{
    Metric metric = get_metric(METRIC_L2, VectorType::VECTOR_FP32);
    for (size_t m = 0; m < M; ++m) {
        size_t idxm = 0;
        float  dism = std::numeric_limits<float>::max();
        const float* xsub = x + m * dsub;
        for(size_t i = 0; i < ksub; ++i) {
            float dis = metric(xsub, get_centroids(m, i), dsub * sizeof(float));
            if (dism > dis) {
                dism = dis;
                idxm = i;
            }
        }
        code[m] = idxm;
    }
}

void ProductQuantizer::compute_l2_distance_table(const float* x, float* dis_table) const
{
}
    
void ProductQuantizer::compute_ip_distance_table(const float* x, float* dis_table) const
{
}

VATTN_NAMESPACE_END