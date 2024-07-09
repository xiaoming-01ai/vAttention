#pragma once

#include "vattn.h"
#include "common/vattn_assert.h"

VATTN_NAMESPACE_BEGIN

enum VectorType {
    VECTOR_BF16 = 0, 
    VECTOR_FP32 = 1,
};

inline int get_data_size(VectorType vtype)
{
    switch(vtype) {
    case VECTOR_BF16:
        return 2;
    case VECTOR_FP32:
        return 4;
    default:
        VATTN_THROW_MSG("vector type only support VECTOR_BF16/VECTOR_FP32.");
    }
}

inline int get_vector_size(VectorType vtype, int d)
{
    return d * get_data_size(vtype);
}

VATTN_NAMESPACE_END