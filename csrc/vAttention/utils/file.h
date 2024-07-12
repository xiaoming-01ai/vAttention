#pragma once

#include "vattn.h"

VATTN_NAMESPACE_BEGIN



int64_t get_file_size(const char *file_path);
int64_t get_file_size(const std::string &file_path);
void get_file_content(const std::string &file_path, std::string &content);
std::string get_file_content(const std::string &file_path);









VATTN_NAMESPACE_END