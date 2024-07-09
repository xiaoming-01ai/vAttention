#include "kvcache/index/metric_type.h"
#include <immintrin.h>

VATTN_NAMESPACE_BEGIN

template <typename T>
inline float l2_metric_typed(const T *lhs, const T *rhs, size_t size)
{
    const T *end = lhs + size;
    const T *alignedEnd = lhs + ((size >> 3) << 3);
    float res = 0.0f;
    for (; lhs != alignedEnd; lhs += 8, rhs += 8) {
        float d0 = lhs[0] - rhs[0];
        float d1 = lhs[1] - rhs[1];
        float d2 = lhs[2] - rhs[2];
        float d3 = lhs[3] - rhs[3];
        float d4 = lhs[4] - rhs[4];
        float d5 = lhs[5] - rhs[5];
        float d6 = lhs[6] - rhs[6];
        float d7 = lhs[7] - rhs[7];
        res += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + \
               d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }
    // fall through all
    float diff = 0.0f;
    switch (end - alignedEnd) {
    case 7:
        diff = lhs[6] - rhs[6];
        res += diff * diff;
    case 6:
        diff = lhs[5] - rhs[5];
        res += diff * diff;
    case 5:
        diff = lhs[4] - rhs[4];
        res += diff * diff;
    case 4:
        diff = lhs[3] - rhs[3];
        res += diff * diff;
    case 3:
        diff = lhs[2] - rhs[2];
        res += diff * diff;
    case 2:
        diff = lhs[1] - rhs[1];
        res += diff * diff;
    case 1:
        diff = lhs[0] - rhs[0];
        res += diff * diff;
    }

    return res;
}

template <typename T>
inline float ip_metric_typed(const T *lhs, const T *rhs, size_t size)
{
    const T *end = lhs + size;
    const T *aligned_end = lhs + ((size >> 3) << 3);
    float res = 0.0f;
    for (; lhs != aligned_end; lhs += 8, rhs += 8) {
        res += (float)(lhs[0]) * rhs[0] + (float)(lhs[1]) * rhs[1] 
             + (float)(lhs[2]) * rhs[2] + (float)(lhs[3]) * rhs[3] 
             + (float)(lhs[4]) * rhs[4] + (float)(lhs[5]) * rhs[5] 
             + (float)(lhs[6]) * rhs[6] + (float)(lhs[7]) * rhs[7];
    }
    // fall through all
    switch (end - aligned_end) {
    case 7:
        res += (float)(lhs[6]) * rhs[6];
    case 6:
        res += (float)(lhs[5]) * rhs[5];
    case 5:
        res += (float)(lhs[4]) * rhs[4];
    case 4:
        res += (float)(lhs[3]) * rhs[3];
    case 3:
        res += (float)(lhs[2]) * rhs[2];
    case 2:
        res += (float)(lhs[1]) * rhs[1];
    case 1:
        res += (float)(lhs[0]) * rhs[0];            
    }

    return res;
}


template <>
inline float l2_metric_typed<float>(const float *lhs, const float *rhs, size_t size)
{
    const float *end = lhs + size;
    const float *alignedEnd = lhs + ((size >> 4) << 4);
    __m512 sum512 = _mm512_setzero_ps();
    for (; lhs != alignedEnd; lhs += 16, rhs += 16) {
        __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
	    sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    }
    float res = _mm512_reduce_add_ps(sum512);

    alignedEnd = lhs + (((end - lhs) >> 3) << 3);
    if (lhs != alignedEnd) {
        __m256 sum256 = _mm256_setzero_ps();
        for (; lhs != alignedEnd; lhs += 8, rhs += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(lhs), 
                                        _mm256_loadu_ps(rhs));
            sum256 = _mm256_fmadd_ps(diff, diff, sum256);
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), 
                                   _mm256_extractf128_ps(sum256, 1));
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum128);
        res += (f[0] + f[1] + f[2] + f[3]);
    }

    float diff = 0.0f;
    switch (end - lhs) {
    case 7:
        diff = lhs[6] - rhs[6];
        res += diff * diff;
    case 6:
        diff = lhs[5] - rhs[5];
        res += diff * diff;
    case 5:
        diff = lhs[4] - rhs[4];
        res += diff * diff;
    case 4:
        diff = lhs[3] - rhs[3];
        res += diff * diff;
    case 3:
        diff = lhs[2] - rhs[2];
        res += diff * diff;
    case 2:
        diff = lhs[1] - rhs[1];
        res += diff * diff;
    case 1:
        diff = lhs[0] - rhs[0];
        res += diff * diff;
    }

    return res;
};

template <>
inline float ip_metric_typed<float>(const float *lhs, const float *rhs, size_t size)
{
    const float *end = lhs + size;
    const float *aligned_end = lhs + ((size >> 4) << 4);
    __m512 sum512 = _mm512_setzero_ps();
    for (; lhs != aligned_end; lhs += 16, rhs += 16) {
	    sum512 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs), sum512);
    }
    float res = _mm512_reduce_add_ps(sum512);
    float diff = 0.0f;
    switch (end - lhs) {
    case 7:
        diff = lhs[6] - rhs[6];
        res += diff * diff;
    case 6:
        diff = lhs[5] - rhs[5];
        res += diff * diff;
    case 5:
        diff = lhs[4] - rhs[4];
        res += diff * diff;
    case 4:
        diff = lhs[3] - rhs[3];
        res += diff * diff;
    case 3:
        diff = lhs[2] - rhs[2];
        res += diff * diff;
    case 2:
        diff = lhs[1] - rhs[1];
        res += diff * diff;
    case 1:
        diff = lhs[0] - rhs[0];
        res += diff * diff;
    }

    return res;
}

inline float l2_metric(const float *lhs, const float *rhs, size_t size)
{
    return l2_metric_typed(lhs, rhs, size);
}

inline float ip_metric(const float *lhs, const float *rhs, size_t size)
{
    return 1.0f - ip_metric_typed(lhs, rhs, size);
}

class FP32L2Metric {
public:
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return l2_metric(static_cast<const float *>(lhs), 
                         static_cast<const float *>(rhs), 
                         size >> 2);
    }
};

class FP32IPMetric {
public:
    float operator()(const void *lhs, const void *rhs, size_t size) const
    {
        return ip_metric(static_cast<const float *>(lhs), 
                         static_cast<const float *>(rhs), 
                         size >> 2);
    }
};

Metric get_ip_metric(VectorType vtype)
{
    switch (vtype) {
    case VECTOR_FP32:
        return FP32IPMetric(); 
    default:
        VATTN_THROW_MSG("ip metric only support vector type 'VECTOR_BF16/VECTOR_FP32'");
    }
}

Metric get_l2_metric(VectorType vtype)
{
    switch (vtype) {
    case VECTOR_FP32:
        return FP32L2Metric(); 
    default:
        VATTN_THROW_MSG("ip metric only support vector type 'VECTOR_FP32'");
    }
}

Metric get_metric(MetricType mtype, VectorType vtype)
{
    switch (mtype) {
    case METRIC_IP:
        return get_ip_metric(vtype);
    case METRIC_L2:
        return get_l2_metric(vtype);
    default:
        VATTN_THROW_MSG("only support metric type 'METRIC_IP/METRIC_L2'");
    }
}

VATTN_NAMESPACE_END