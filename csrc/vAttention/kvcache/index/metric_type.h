#pragma once

#include "kvcache/index/vector_type.h"

#include <functional>

VATTN_NAMESPACE_BEGIN

/// The metric space for vector comparison for indices and algorithms.
/// Most algorithms support both inner product and L2.
enum MetricType {
    METRIC_IP = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
};

using Metric = std::function<float(const void *, const void *, size_t)>;
Metric get_metric(MetricType mtype, VectorType vtype);


VATTN_NAMESPACE_END
