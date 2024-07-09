#pragma once 

#include "vattn.h"
#include <vector>

VATTN_NAMESPACE_BEGIN

/** Class for the clustering parameters. Can be passed to the 
 *  constructor of the Clustering object.
 */
struct ClusteringParameters {
    /// number of clustering iterations
    int niter = 25;

    /// If fewer than this number of training vectors per centroid are provided,
    /// writes a warning. Note that fewer than 1 point per centroid raises an exception.
    int min_points_per_centroid = 16;

    /// to limit size of dataset, otherwise the training set is subsampled
    int max_points_per_centroid = 256;

    /// seed for the random number generator
    int seed = 1234;
};

/** K-means clustering based on assignment - centroid update iterations
 *
 * The clustering is based on an Index object that assigns training
 * points to the centroids. Therefore, at each iteration the centroids
 * are added to the index.
 *
 * On output, the centoids table is set to the latest version
 * of the centroids and they are also added to the index. If the
 * centroids table it is not empty on input, it is also used for
 * initialization.
 *
 */
struct Clustering : ClusteringParameters {
    int d; ///< dimension of the vectors
    int k; ///< nb of centroids

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as initialization
     */
    std::vector<float> centroids;
    
    Clustering(int d, int k);

    /** run k-means training
     *
     * @param x          training vectors, size n * d
     */
    virtual void train(int n, const float* x);

    virtual ~Clustering() {}
};

VATTN_NAMESPACE_END