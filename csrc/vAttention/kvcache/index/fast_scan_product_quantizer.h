#pragma once

#include "vattn.h"
#include "kvcache/index/product_quantizer.h"

VATTN_NAMESPACE_BEGIN

struct FastScanProductQuantizer : ProductQuantizer {
    /// Transposed centroid table, size M * ksub * dsub.
    /// Layout: (M, dsub, ksub)
    std::vector<float> transposed_centroids;

    FastScanProductQuantizer(
        size_t d,      /* dimensionality of the input vectors */
        size_t M);     /* number of subquantizers */

    // Train the product quantizer on a set of points. A clustering
    // can be set on input to define non-default clustering parameters
    void train(size_t n, const float *x) override;

    void _compute_code(const float* x, uint16_t* codes) const;
    /** Quantize a set of vectors
     *
     * @param x        input vectors, size n * d
     * @param codes    output codes, size n * code_size
     */
    void compute_codes(size_t n, const float* x, uint8_t* codes) const override;

    void compute_codes_pack(size_t n, const float *x, uint8_t *codes) const;

    // void compute_float_LUT(float* lut, idx_t n, const float* x) const override;

    // called by search function
    void compute_quantized_ip_lut(size_t n, const float* x, uint8_t* lut) const;
    void compute_quantized_l2_lut(size_t n, const float* x, uint8_t* lut) const;
    
    void compute_quantized_ip_lut(const float* x, uint8_t* lut) const;
    void compute_quantized_l2_lut(const float* x, uint8_t* lut) const;

    /** Compute distance table for one vector.
     *
     * The distance table for x = [x_0 x_1 .. x_(M-1)] is a M * ksub
     * matrix that contains
     *
     *   dis_table (m, j) = || x_m - c_(m, j)||^2
     *   for m = 0..M-1 and j = 0 .. ksub - 1
     *
     * where c_(m, j) is the centroid no j of sub-quantizer m.
     *
     * @param x         input vector size d
     * @param dis_table output table, size M * ksub
     */
    // void compute_l2_distance_table(const float* x, float* dis_table) const override;
    // void compute_ip_distance_table(const float* x, float* dis_table) const override;

    void compute_distance(const uint8_t *code, const uint8_t *lut, int32_t *distance) const;

    /// Sync transposed centroids with regular centroids. This call is needed if centroids 
    /// were edited directly.
    /// Transpose Layout:(M, ksub, dsub) to Layout:(M, dsub, ksub)
    void sync_transposed_centroids();

    /// Clear transposed centroids table so ones are no longer used.
    void clear_transposed_centroids();
};
using FastScanProductQuantizerPtr = std::shared_ptr<FastScanProductQuantizer>;

VATTN_NAMESPACE_END