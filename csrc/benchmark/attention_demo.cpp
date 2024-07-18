// #include "vattn.h"
#include "attention.h"
#include "common/vattn_assert.h"
#include "utils/file.h"
#include "utils/util.h"

VATTN_NAMESPACE_USE

int main(int argc, char *argv[])
{
    int kv_head_cnt = 4;
    int kv_head_dim = 128;
    int q_head_cnt = 32;
    int q_head_dim = 128;
    int seqs_len = 513;

    int batch_size = 1;

    const char *k_file = "/xpfs/public/xiaoming/vllm-infra/datas/attention_3/k.bin.step1";
    const char *v_file = "/xpfs/public/xiaoming/vllm-infra/datas/attention_3/v.bin.step1";
    const char *q_file = "/xpfs/public/xiaoming/vllm-infra/datas/attention_3/q.bin.step1";

    std::string k_str = get_file_content(k_file);
    // fprintf(stderr, "K cache: load %lu bytes from file %s\n", k_str.size(), k_file);
    size_t k_nbytes = sizeof(float) * kv_head_dim * seqs_len * kv_head_cnt;
    VATTN_ASSERT_FMT(k_str.size() == k_nbytes, "%lu != %lu", k_str.size(), k_nbytes);
    fprintf(stderr, "Load K(%d, %d, %d)\n", seqs_len, kv_head_cnt, kv_head_dim);


    std::string v_str = get_file_content(v_file);
    // fprintf(stderr, "V cache: load %lu bytes from file %s\n", v_str.size(), v_file);
    size_t v_nbytes = sizeof(float) * kv_head_dim * seqs_len * kv_head_cnt;
    VATTN_ASSERT_FMT(v_str.size() == v_nbytes, "%lu != %lu", v_str.size(), v_nbytes);
    fprintf(stderr, "Load V(%d, %d, %d)\n", seqs_len, kv_head_cnt, kv_head_dim);
    
    std::string q_str = get_file_content(q_file);
    // fprintf(stderr, "Q cache: load %lu bytes from file %s\n", q_str.size(), q_file);
    size_t q_nbytes = sizeof(float) * q_head_dim * q_head_cnt;
    VATTN_ASSERT_FMT(q_str.size() == q_nbytes, "%lu != %lu", q_str.size(), q_nbytes);
    fprintf(stderr, "Load Q(%d, %d, %d)\n", 1, q_head_cnt, q_head_dim);

    Attention attn("FLAT") ;

    float *d_k;
    cudaMalloc((void**)&d_k, k_nbytes);
    cudaMemcpy(d_k, k_str.data(), k_nbytes, cudaMemcpyHostToDevice);
    
    float *d_v;
    cudaMalloc((void**)&d_v, v_nbytes);
    cudaMemcpy(d_v, v_str.data(), v_nbytes, cudaMemcpyHostToDevice);

    attn.cache_fp32(d_k, d_v, seqs_len, kv_head_cnt, kv_head_dim, 4096, 0);

    std::vector<float> output(q_head_cnt * q_head_dim, 0);
    float scale = 0.08838834764831845f;
    attn.forward_fp32((const float *)q_str.data(), q_head_cnt, q_head_dim, 1024, scale, output.data(), 0);

    print_value<float>("output:", output.data(), q_head_dim, 16);
    
    return 0;
}