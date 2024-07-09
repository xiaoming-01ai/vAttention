#include "kvcache/index/fast_scan_product_quantizer.h"
// #include "utils/util.h"

#include <immintrin.h>
#include <stdlib.h>
#include <algorithm>
#include <tgmath.h>

VATTN_NAMESPACE_BEGIN

// there can be NaNs in tables, they should be ignored
static void tab_min_max(const float* tab, size_t n, float &min, float &max) 
{
    min = tab[0];
    max = tab[0];
    for (size_t i = 1; i < n; i++) {
        auto val = tab[i];
        if (val < min) {
            min = val;
        }
        if (val > max) {
            max = val;
        }
    }
}

static void round_tab(float* tab, size_t n, float a, float bi) 
{
    for (size_t i = 0; i < n; i++) {
        tab[i] = floorf((tab[i] - bi) * a + 0.5);
    }
}

static void round_uint8_per_column(float* tab, size_t n, size_t d) {
    float max_span = 0;
    std::vector<float> mins(n);
    float max = 0;
    for (size_t i = 0; i < n; i++) {
        tab_min_max(tab + i * d, d, mins[i], max);
        float span = max - mins[i];
        if (span > max_span) {
            max_span = span;
        }
    }
    float a = 255 / max_span;
    for (size_t i = 0; i < n; i++) {
        round_tab(tab + i * d, d, a, mins[i]);
    }
}

static void ip_kernel(size_t dsub, size_t ksub, const float *lhs, const float *rhs, float *distance) 
{
    __m512 ret512 = _mm512_setzero_ps();
    for (size_t j = 0; j < dsub; ++j) {
        auto lhs512 = _mm512_set1_ps(lhs[j]);
        ret512 = _mm512_fmadd_ps(lhs512, _mm512_loadu_ps(rhs + j * ksub), ret512);
    }
    ret512 = _mm512_sub_ps(_mm512_set1_ps(1), ret512);
    _mm512_store_ps(distance, ret512);
}

static void l2_kernel(size_t dsub, size_t ksub, const float *lhs, const float *rhs, float *distance) 
{
    __m512 ret512 = _mm512_setzero_ps();
    for (size_t j = 0; j < dsub; ++j) {
        auto lhs512 = _mm512_set1_ps(lhs[j]);
        __m512 diff = _mm512_sub_ps(lhs512, _mm512_loadu_ps(rhs + j * ksub));
        ret512 = _mm512_fmadd_ps(diff, diff, ret512);
    }
    _mm512_store_ps(distance, ret512);
}

FastScanProductQuantizer::FastScanProductQuantizer(size_t d, size_t M)
    : ProductQuantizer(d, M, 4)
{ }

void FastScanProductQuantizer::train(size_t n, const float *x)
{
    ProductQuantizer::train(n, x);
    sync_transposed_centroids();
}

void FastScanProductQuantizer::compute_codes_pack(size_t n, const float *x, uint8_t *codes) const
{
    static constexpr int idx[] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    thread_local std::vector<uint16_t> code;
    code.resize(M);

    for (size_t i = 0; i < 16 && i < n; ++i) {
        _compute_code(x + d * idx[i], code.data());
        uint8_t *ptr = codes + i;
        for (size_t j = 0; j < M; ++j) {
            *(ptr + j * 16) = 0xF0 & (code[j] << 4);
        }
    }

    for (size_t i = 0; i < 16 && i < n - 16; ++i) {
        _compute_code(x + d * (16 + idx[i]), code.data());
        uint8_t *ptr = codes + i;
        for (size_t j = 0; j < M; ++j) {
            *(ptr + j * 16) |= 0x0F & code[j];
        }
    }
}

// void FastScanProductQuantizer::compute_codes_pack(size_t n, const float *x, uint8_t *codes)
// {

// }
    
void FastScanProductQuantizer::_compute_code(const float* x, uint16_t* codes) const
{
    float distance[16];
    for (size_t mi = 0; mi < M; ++mi) {
        const float* centrois_ptr = transposed_centroids.data() + mi * ksub * dsub;
        const float* x_ptr = x + mi * dsub;

        __m512 ret512 = _mm512_setzero_ps();
        for (size_t di = 0 ; di < dsub; ++di) {
            __m512 diff =  _mm512_sub_ps(_mm512_set1_ps(x_ptr[di]), 
                                         _mm512_loadu_ps(centrois_ptr + di * ksub));
            ret512 = _mm512_fmadd_ps(diff, diff, ret512);
        }
        _mm512_store_ps(distance, ret512);

        float  dism = distance[0];
        size_t idxm = 0;
        for (size_t j = 1; j < 16; ++j) {
            if (distance[j] < dism) {
                dism = distance[j];
                idxm = j;
            }
        }
        codes[mi] = idxm;
    }
}

void FastScanProductQuantizer::compute_codes(size_t n, const float *x, uint8_t *codes) const
{
    size_t ne = (n >> 5) << 5;
// #pragma omp parallel for num_threads(6)
    #pragma omp parallel for
    for (size_t i = 0; i < ne; i += 32) {
        compute_codes_pack(32, x + i * d, codes + i * code_size);
    }
    if (n - ne > 0) {
        std::vector<float> tmp(32 * d);
        memcpy(tmp.data(), x + ne * d, sizeof(float) * (n - ne) * d);
        compute_codes_pack(n - ne, tmp.data(), codes + ne * code_size);
    }
}

void FastScanProductQuantizer::compute_quantized_ip_lut(const float* x, uint8_t* lut) const
{
    std::shared_ptr<float[]> distance((float *)aligned_alloc(64,  M * ksub * sizeof(float)), free);
    for (size_t m = 0; m < M; ++m) {
        ip_kernel(dsub, 
                  ksub, 
                  x + m * dsub, 
                  transposed_centroids.data() + m * ksub * dsub, 
                  distance.get() + m * ksub); 
    }
    
    round_uint8_per_column(distance.get(), M, ksub);
    size_t cnt = M * ksub;
    for (size_t i = 0; i < cnt; ++i) {
        lut[i] = int(distance[i]);
    }
}

void FastScanProductQuantizer::compute_quantized_l2_lut(const float* x, uint8_t* lut) const
{
    std::shared_ptr<float[]> distance((float *)aligned_alloc(64,  M * ksub * sizeof(float)), free);
    for (size_t m = 0; m < M; ++m) {
        l2_kernel(dsub, 
                  ksub, 
                  x + m * dsub, 
                  transposed_centroids.data() + m * ksub * dsub, 
                  distance.get()); 
    }
    round_uint8_per_column(distance.get(), M, ksub);
    size_t cnt = M * ksub;
    for (size_t i = 0; i < cnt; ++i) {
        lut[i] = int(distance[i]);
    }
}
    
void FastScanProductQuantizer::compute_quantized_ip_lut(size_t n, const float* x, uint8_t* lut) const
{
#pragma omp parallel for
    for (size_t ni = 0; ni < n; ++ni) {
        compute_quantized_ip_lut(x + ni * d, &lut[ni * ksub * M]);
    }
}

void FastScanProductQuantizer::compute_quantized_l2_lut(size_t n, const float* x, uint8_t* lut) const
{
    std::unique_ptr<float[]> dis_tables(new float[n * ksub * M]);

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        compute_l2_distance_table(x + i * d, &dis_tables[i * ksub * M]);
    }
}
    
void FastScanProductQuantizer::compute_distance(
    const uint8_t *codes, const uint8_t *lut, int32_t *distances) const
{
    static constexpr size_t AVX512_BYTES_STRIDE = 64;
    __m512i mask512x0F = _mm512_set1_epi8(0x0F);

    const uint8_t *last = codes + 32 * code_size;
    __m512i p0016 = _mm512_setzero_si512();
    __m512i p0116 = _mm512_setzero_si512();
    __m512i p1016 = _mm512_setzero_si512();
    __m512i p1116 = _mm512_setzero_si512();

    while (codes < last)
    {
        __m512i lookup_table = _mm512_loadu_si512((__m512i const *)lut);
        // Prefetch((const char *)&codes[0] + 64 * 8, 1);
        // Prefetch((const char *)distance_table + 64 * 8, 1);

        __m512i packedobj = _mm512_loadu_si512((__m512i const *)codes);
        
        // first 16vec
        __m512i packedobj_h = _mm512_and_si512(_mm512_srli_epi16(packedobj, 4), mask512x0F); 
        // last 16 vec
        __m512i packedobj_l = _mm512_and_si512(packedobj, mask512x0F); 

        __m512i vtmp_l = _mm512_shuffle_epi8(lookup_table, packedobj_l);
        __m512i vtmp_h = _mm512_shuffle_epi8(lookup_table, packedobj_h);

        p0016 = _mm512_add_epi16(p0016, vtmp_l);  // 全量按照16位累加
        p1016 = _mm512_add_epi16(p1016, vtmp_h);
        
        // 偶数位 距离累加 0 2 4 6 8 10 12 14 16.... 28 30
        p0116 = _mm512_add_epi16(p0116, _mm512_srli_epi16(vtmp_l, 8)); 
        p1116 = _mm512_add_epi16(p1116, _mm512_srli_epi16(vtmp_h, 8));
    
        codes += AVX512_BYTES_STRIDE;
        lut   += AVX512_BYTES_STRIDE;
    }

    // 奇数位 距离累加 1 3 5 7 9.....29 31. 
    p0016 = _mm512_sub_epi16(p0016, _mm512_slli_epi16(p0116, 8)); 
    p1016 = _mm512_sub_epi16(p1016, _mm512_slli_epi16(p1116, 8));

    // 16 个 int32, [0 2 4 6 8 10 12 14] [1 3 5 7 9 11 13 15] []
    __m512i a0b1a2b3 = _mm512_mask_blend_epi32(0xF0F0, p0016, p0116);
    __m512i c0d1c2d3 = _mm512_mask_blend_epi32(0XF0F0, p1016, p1116);
    
    __m512i mask = _mm512_set_epi32(27, 26, 25, 24, 15, 14, 13, 12, 19, 18, 17, 16, 7, 6, 5, 4);
    __m512i a1b0a3b2 = _mm512_permutex2var_epi32(p0016, mask, p0116);
    __m512i c1d0c3d2 = _mm512_permutex2var_epi32(p1016, mask, p1116);

    __m512i tem = _mm512_adds_epu16(c0d1c2d3, c1d0c3d2);
    __m512i lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(tem, 0));
    __m512i  hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(tem, 1));
    __m512i distacne_1 = _mm512_add_epi32(lo, hi);
    _mm512_store_epi32(distances, distacne_1);
    distances += 16;

    tem = _mm512_adds_epu16(a0b1a2b3, a1b0a3b2);
    lo  = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(tem, 0));
    hi  = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(tem, 1));
    __m512i distacne_0 = _mm512_add_epi32(lo, hi);
    _mm512_store_epi32(distances, distacne_0);
    distances += 16;
}

void FastScanProductQuantizer::sync_transposed_centroids() 
{
    // Transpose Layout:(M, ksub, dsub) to Layout:(M, dsub, ksub)
    transposed_centroids.resize(M * dsub * ksub);
    for (size_t mi = 0; mi < M; ++mi) {
        for (size_t di = 0; di < dsub; ++di) {
            for (size_t ki = 0; ki < ksub; ++ki) {
                transposed_centroids[mi * dsub * ksub + di * ksub + ki] = 
                    centroids[mi * dsub * ksub + ki * dsub + di];
            }
        }
    }
}

void FastScanProductQuantizer::clear_transposed_centroids() 
{
    transposed_centroids.clear();
    transposed_centroids.shrink_to_fit();
}

VATTN_NAMESPACE_END
