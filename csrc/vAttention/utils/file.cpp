#include "utils/file.h"
#include "common/vattn_assert.h"

#include <sys/stat.h>

VATTN_NAMESPACE_BEGIN

int64_t get_file_size(const char *file_path)
{
    if (file_path == nullptr) {
        return -1;
    }

    struct stat info;
    int ret = stat(file_path, &info);
    if (ret != 0) {
        return -1;
    }
    return info.st_size;
}

int64_t get_file_size(const std::string &file_path)
{
    return get_file_size(file_path.c_str());
}

void get_file_content(const std::string &file_path, std::string &content)
{
    FILE* file = fopen(file_path.c_str(), "rb");
    VATTN_THROW_IF_NOT_FMT(file != nullptr,
                           "Open file[%s] for read failed.",
                           file_path.c_str());

    size_t file_size = get_file_size(file_path);
    content.resize(file_size);
    size_t ret = fread((void *)content.data(), sizeof(char), file_size, file);
    fclose(file);

    VATTN_THROW_IF_NOT_FMT(ret == file_size,
                           "Get file content failed. File orignal size[%lu]. readed size[%lu]",
                           file_size, ret);
}

std::string get_file_content(const std::string &file_path)
{
    FILE* file = fopen(file_path.c_str(), "rb");
    VATTN_THROW_IF_NOT_FMT(file != nullptr,
                           "Open file[%s] for read failed.",
                           file_path.c_str());

    size_t file_size = get_file_size(file_path);
    std::string content(file_size, 0);
    size_t ret = fread((void *)content.data(), sizeof(char), file_size, file);
    fclose(file);

    VATTN_THROW_IF_NOT_FMT(ret == file_size,
                           "Get file content failed. File orignal size[%lu]. readed size[%lu]",
                           file_size, ret);
    return content;
}

VATTN_NAMESPACE_END