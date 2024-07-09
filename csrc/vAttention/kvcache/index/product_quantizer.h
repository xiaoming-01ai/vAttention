#pragma once

#include <stdint.h>
#include <vector>

#include "kvcache/index/clustering.h"
#include "kvcache/index/quantizer.h"

VATTN_NAMESPACE_BEGIN

/** Product Quantizer. Implemented only for METRIC_L2 */
struct ProductQuantizer : Quantizer {
    size_t M;        ///< number of subquantizers
    size_t nbits;    ///< number of bits per quantization index

    // values derived from the above
    size_t dsub;     ///< dimensionality of each subvector
    size_t ksub;     ///< number of centroids for each subquantizer

    /// Centroid table, size M * ksub * dsub.
    /// Layout: (M, ksub, dsub)
    std::vector<float> centroids;
    
    ProductQuantizer(size_t d,      /* dimensionality of the input vectors */
                     size_t M,      /* number of subquantizers */
                     size_t nbits); /* number of bit per subvector index */

    /// return the centroids associated with subvector m
    float* get_centroids(size_t m, size_t i) {
        return &centroids[(m * ksub + i) * dsub];
    }

    const float* get_centroids(size_t m, size_t i) const {
        return &centroids[(m * ksub + i) * dsub];
    }

    // Train the product quantizer on a set of points. A clustering
    // can be set on input to define non-default clustering parameters
    void train(size_t n, const float* x) override;

    /// Quantize one vector with the product quantizer
    void compute_code(const float* x, uint16_t* code) const;

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
    virtual void compute_l2_distance_table(const float* x, float* dis_table) const;
    virtual void compute_ip_distance_table(const float* x, float* dis_table) const;

    /** compute distance table for several vectors
     * @param nx        nb of input vectors
     * @param x         input vector size nx * d
     * @param dis_table output table, size nx * M * ksub
     */
    void compute_l2_distance_tables(size_t nx, const float* x, float* dis_tables) const;
    void compute_ip_distance_tables(size_t nx, const float* x, float* dis_tables) const;
};

VATTN_NAMESPACE_END