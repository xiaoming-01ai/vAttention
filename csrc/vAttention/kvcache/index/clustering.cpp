#include "common/vattn_assert.h"
#include "kvcache/index/clustering.h"
#include "kvcache/index/vector_type.h"
#include "kvcache/index/metric_type.h"
#include "utils/random.h"
#include "utils/timer.h"

#include <omp.h>
#include <immintrin.h>

VATTN_NAMESPACE_BEGIN

Clustering::Clustering(int d, int k) : d(d), k(k) {}

using LocalMetric = std::function<float(const float *, const float *)>;

inline float l2_metric_d1(const float *lhs, const float *rhs)
{
    float dif0 = lhs[0] - rhs[0];
    return dif0 * dif0;
};

inline float l2_metric_d2(const float *lhs, const float *rhs)
{
    float dif0 = lhs[0] - rhs[0];
    float dif1 = lhs[1] - rhs[1];
    return dif0 * dif0 + dif1 * dif1;
};

inline float l2_metric_d4(const float *lhs, const float *rhs)
{
    float dif0 = lhs[0] - rhs[0];
    float dif1 = lhs[1] - rhs[1];
    float dif2 = lhs[2] - rhs[2];
    float dif3 = lhs[3] - rhs[3];
    return dif0 * dif0 + dif1 * dif1 + dif2 * dif2 + dif3 * dif3;
};

inline float l2_metric_d8(const float *lhs, const float *rhs)
{
    float dif0 = lhs[0] - rhs[0];
    float dif1 = lhs[1] - rhs[1];
    float dif2 = lhs[2] - rhs[2];
    float dif3 = lhs[3] - rhs[3];
    float dif4 = lhs[4] - rhs[4];
    float dif5 = lhs[5] - rhs[5];
    float dif6 = lhs[6] - rhs[6];
    float dif7 = lhs[7] - rhs[7];
    return dif0 * dif0 + dif1 * dif1 + dif2 * dif2 + dif3 * dif3
         + dif4 * dif4 + dif5 * dif5 + dif6 * dif6 + dif7 * dif7;
};

inline float l2_metric_d16(const float *lhs, const float *rhs)
{
    __m512 sum512 = _mm512_setzero_ps();
    __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
	sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    return _mm512_reduce_add_ps(sum512);
};

inline float l2_metric_d32(const float *lhs, const float *rhs)
{
    __m512 sum512 = _mm512_setzero_ps();
    __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
	sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    diff = _mm512_sub_ps(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16));
    return _mm512_reduce_add_ps(sum512);
};

LocalMetric get_local_metric(int d)
{
    switch (d) {
    case 1:
        return l2_metric_d1;
    case 2:
        return l2_metric_d2;
    case 4:
        return l2_metric_d4;
    case 8:
        return l2_metric_d8;
    case 16:
        return l2_metric_d16;
    default:
        return nullptr;
    }
}

void assign_centroid(int d,
                     int k,
                     int n,
                     const char *x,
                     int *assign,
                     const float *centroids) 
{
    // size_t line_size = d * sizeof(float);
    auto metric = get_local_metric(d);
#pragma omp parallel for num_threads(6)
    for (int i = 0; i < n; ++i) {
        const float *xi = (const float *)x + (size_t)i * d;
        float min_dis = std::numeric_limits<float>::max();
        int min_cj = 0;
        for (int j = 0; j < k; ++j) {
            float dis = metric(centroids + j * d, xi);
            if (dis < min_dis) {
                min_cj = j;
                min_dis = dis;
            }
        }
        assign[i] = min_cj;
    }
}

/** compute centroids as sum of training points
 *
 * @param x            training vectors, size n * code_size (from codec)
 * @param assign       nearest centroid for each training vector, size n
 * @param centroids    centroid vectors (output only), size k * d
 * @param hassign      histogram of assignments per centroid (size k),
 *                     should be 0 on input
 */
void compute_centroids(
        int d,
        int k,
        int n,
        const char *x,
        const int *assign,
        float *hassign,
        float *centroids) 
{
    memset(centroids, 0, sizeof(*centroids) * d * k);
    memset(hassign, 0, sizeof(float) * k);
    size_t line_size = d * sizeof(float);

#pragma omp parallel num_threads(6)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        int c0 = (k * rank) / nt;
        int c1 = (k * (rank + 1)) / nt;

        for (int i = 0; i < n; i++) {
            int ci = assign[i];
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi = reinterpret_cast<const float*>(x + i * line_size);
                hassign[ci] += 1.0;
                for (int j = 0; j < d; j++) {
                    c[j] += xi[j];
                }
            }
        }
    }

#pragma omp parallel for num_threads(6)
    for (int ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float* c = centroids + ci * d;
        for (int j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)
/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one.
 *
 * @return           nb of spliting operations (larger is worse)
 */
int split_clusters(int d, int k, int n, float* hassign, float* centroids) 
{
    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (int ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            int cj;
            for (cj = 0; true; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy(centroids + ci * d, centroids + cj * d, sizeof(*centroids) * d);

            /* small symmetric pertubation */
            for (int j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}

int subsample_training_set(const Clustering& clus,
                           int nx,
                           const char *x,
                           size_t line_size,
                           char ** x_out) 
{
    // fprintf(stderr, "Sampling a subset of %d / %d for training\n",
    //         clus.k * clus.max_points_per_centroid, nx);
    std::vector<int> perm(nx);
    rand_perm(perm.data(), nx, clus.seed);
    nx = clus.k * clus.max_points_per_centroid;
    char *x_new = new char[nx * line_size];
    *x_out = x_new;
    for (int i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }
    return nx;
}

void Clustering::train(int nx, const float* x_in)
{
    VATTN_THROW_IF_NOT_FMT(
            nx >= k,
            "Number of training points %d should be at least as large as number of clusters (%d)",
            nx, k);

    const char *x = (const char *)x_in;
    std::unique_ptr<char[]> del1;
    std::unique_ptr<float[]> del3;
    size_t line_size = sizeof(float) * d;

    if (nx > k * max_points_per_centroid) {
        char * x_new;
        nx = subsample_training_set(*this, nx, x, line_size, &x_new);
        del1.reset(x_new);
        x = x_new;
    } else if (nx < k * min_points_per_centroid) {
        printf("WARNING clustering %d points to %d centroids:"
               " please provide at least %d training points\n",
               nx, k, k * min_points_per_centroid);
    }

     if (nx == k) {
         // this is a corner case, just copy training set to clusters
         printf("Number of training points %d same as number of clusters, just copying\n", nx);
         
         centroids.resize(d * k);
         memcpy(centroids.data(), x_in, sizeof(float) * d * k);
         return;
    }

    // printf("Clustering %d points in %dD to %d clusters, %d iterations\n", nx, d, k, niter);

    centroids.resize(d * k);
    std::vector<int> perm(nx);
    rand_perm(perm.data(), nx, seed + 1 + 15486557L);
    for (int i = 0; i < k; i++) {
        memcpy(&centroids[i * d], x + perm[i] * line_size, line_size);
    }

    std::unique_ptr<int[]> assign(new int[nx]);
    std::unique_ptr<float[]> dis(new float[nx]);

    // k-means iterations
    std::vector<float> hassign(k);
    for (int i = 0; i < niter; i++) {
        // auto t0s = Timer::get_millisecs();

        assign_centroid(d, k, nx, x, assign.get(), centroids.data());

        // update the centroids
        compute_centroids(d, k, nx, x, assign.get(), hassign.data(), centroids.data());
        split_clusters(d, k, nx, hassign.data(), centroids.data());
    }
}

VATTN_NAMESPACE_END