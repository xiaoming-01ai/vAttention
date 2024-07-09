#include "attention.h"

VATTN_NAMESPACE_BEGIN

bool Attention::cache(const void *key, const void *value, int kv_cnt)
{
    return false;
}
    
bool Attention::forward_prefix(const void *query)
{
    return false;
}

bool Attention::forward_decode(const void *query)
{
    
    return false;
}

VATTN_NAMESPACE_END