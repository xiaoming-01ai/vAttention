// #pragma once

// #include "kvcache/index/fast_scan_product_quantizer.h"
// #include "kvcache/index/cache_index.h"
// #include "utils/aligned_table.h"

// VATTN_NAMESPACE_BEGIN

// /** Fast scan version of IndexPQ. Works for 4-bit PQ for now.
//  *
//  * The codes are not stored sequentially but grouped in blocks of size 32.
//  * This makes it possible to compute distances quickly with SIMD instructions.
//  *
//  */

// struct CacheIndexFastScanPQ : CacheIndex {
//     std::vector<FastScanProductQuantizerPtr> pq;

//     AlignedTable<uint8_t> codes;

//     int M;

//     /**
//      * @param d dimensionality of the input vectors
//      * @param M number of subquantizers
//      */
//     CacheIndexFastScanPQ(int d, int M);

//     void bind_fp32(int heads, int seqs, const void **keys) override;

//     int search(int heads, const float **q, int k, int **labels) const override;
// };

// VATTN_NAMESPACE_END
