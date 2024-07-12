#include "vattn.h"
#include "utils/file.h"
#include "common/vattn_assert.h"

VATTN_NAMESPACE_USE

int main(int argc, char *argv[])
{
    int kv_head_cnt = 4;
    int kv_head_dim = 128;
    int q_head_cnt = 32;
    int q_head_dim = 128;
    int seqs_len = 512;

    int batch_size = 1;

    std::string k_str = get_file_content(argv[1]);
    fprintf(stderr, "K cache: load %lu bytes from file %s\n", k_str.size(), argv[1]);
    size_t nbytes = sizeof(float) * kv_head_dim * seqs_len * kv_head_cnt;
    VATTN_ASSERT_FMT(k_str.size() == nbytes, "%lu != %lu", k_str.size(), nbytes);


    std::string v_str = get_file_content(argv[2]);
    fprintf(stderr, "V cache: load %lu bytes from file %s\n", v_str.size(), argv[2]);
    nbytes = sizeof(float) * kv_head_dim * seqs_len * kv_head_cnt;
    VATTN_ASSERT_FMT(v_str.size() == nbytes, "%lu != %lu", v_str.size(), nbytes);
    
    std::string q_str = get_file_content(argv[3]);
    fprintf(stderr, "Q cache: load %lu bytes from file %s\n", q_str.size(), argv[3]);
    nbytes = sizeof(float) * q_head_dim * q_head_cnt;
    VATTN_ASSERT_FMT(q_str.size() == nbytes, "%lu != %lu", q_str.size(), nbytes);
    
    return 0;
}