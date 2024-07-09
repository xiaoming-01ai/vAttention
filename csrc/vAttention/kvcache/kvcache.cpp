#include "kvcache.h"
#include "common/vattn_assert.h"
#include "kvcache/index/cache_index_flat.h"
#include "kvcache/index/cache_index_fast_scan_pq.h"

#include <regex>

VATTN_NAMESPACE_BEGIN

KVCache::KVCache(int head_dim, const std::string &index_desc)
{
    VATTN_THROW_IF_NOT_FMT(head_dim != 0 && head_dim <= 65536,
                            "The head_dim %d shoule be > 0 and < 65536",
                            head_dim);
    
    std::smatch sm;
    auto match = [&sm, index_desc](const std::string& pattern) {
        return std::regex_match(index_desc, sm, std::regex(pattern));
    };

    if (match("FSPQ([0-9]+)")) {
        int M = std::stoi(sm[1].str());
        index.reset(new CacheIndexFastScanPQ(head_dim, M));
    } else if (match("FLAT")) {
        index.reset(new CacheIndexFlat(head_dim));
    } else { 
        VATTN_ASSERT_FMT(false, 
                         "Index description[%s] not support", 
                         index_desc.c_str());
    }
}
    
bool KVCache::bind_fp32(int heads, int seqs, const void **keys) 
{
    this->keys = keys;
    try {
        fprintf(stderr, "kvcache %d %d\n", heads, seqs);
        index->bind_fp32(heads, seqs, this->keys);
        return true;
    } catch (const std::exception &ex) {
        fprintf(stderr, "KVCache bind exception[%s]\n", ex.what());
        return false;
    }
}
    
bool KVCache::bind_bf16(int heads, int seqs, const void **keys)
{
    this->keys = keys;
    try {
        index->bind_bf16(heads, seqs, this->keys);
        return true;
    } catch (const std::exception &ex) {
        fprintf(stderr, "KVCache bind exception[%s]\n", ex.what());
        return false;
    }
}
    
int KVCache::search(int heans, const float **q, int k, int **labels)
{
    try {
        return index->search(heans, q, k, labels);
    } catch (const std::exception &ex) {
        fprintf(stderr, "KVCache bind exception[%s]\n", ex.what());
        return 0;
    }
}
    
VATTN_NAMESPACE_END