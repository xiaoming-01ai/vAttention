#pragma once

#include "vattn.h"

VATTN_NAMESPACE_BEGIN

struct Neighbor {
    int id;
    float score;
    
    Neighbor() = default;
    Neighbor(int input_id, float input_score)
        : id(input_id), score(input_score)
    { }
    
    bool operator<(const Neighbor &rhs) const
    {
        if (score < rhs.score) {
            return true;
        } else if (score > rhs.score) {
            return false;
        } else {
            return id < rhs.id;
        }
    }

    bool operator==(int cmp_id) const
    {
        return id == cmp_id;
    }
};


VATTN_NAMESPACE_END