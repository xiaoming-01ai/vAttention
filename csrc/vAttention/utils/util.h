#pragma once

#include "vattn.h"


VATTN_NAMESPACE_BEGIN

template <typename T>
void print_value(const char *desc, const T *arr, int cnt, int split)
{
    fprintf(stderr, "%s\n", desc);
    for (int i = 0; i < cnt; ++i) {
        if (((i) % split) == 0) {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%f ", (float)arr[i]);
    }
    fprintf(stderr, "\n");
}






VATTN_NAMESPACE_END