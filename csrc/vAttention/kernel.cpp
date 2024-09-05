#include "kernel.h"

#include <immintrin.h>

VATTN_NAMESPACE_BEGIN

void compute_v_fp32(float *dst, const float *src, float scale, int size)
{
    const float *end = src + size;
    const float *aligned_end = src + ((size >> 4) << 4);
    __m512 scale512 = _mm512_set1_ps(scale);
    for (; src != aligned_end; src += 16, dst += 16) {
        auto dst512 = _mm512_fmadd_ps(scale512, _mm512_loadu_ps(src), _mm512_loadu_ps(dst));
	    _mm512_storeu_ps(dst, dst512);
    }

    aligned_end = src + (((end - src) >> 3) << 3);
    if (src != aligned_end) {
        __m256 scale256 = _mm256_set1_ps(scale);
	    _mm256_store_ps(dst, _mm256_fmadd_ps(scale256, _mm256_loadu_ps(src), _mm256_loadu_ps(dst)));
        src += 8;
        dst += 8;
    }

    switch (end - src) {
        case 7:
            dst[7] += (src[7] * scale);
        case 6:
            dst[6] += (src[6] * scale);
        case 5:
            dst[5] += (src[5] * scale);
        case 4:
            dst[4] += (src[4] * scale);
        case 3:
            dst[3] += (src[3] * scale);
        case 2:
            dst[2] += (src[2] * scale);
        case 1:
            dst[1] += (src[1] * scale);
    }
}

VATTN_NAMESPACE_END