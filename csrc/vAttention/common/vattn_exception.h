#pragma once

#include <exception>
#include <string>
#include <utility>
#include <vector>
#include "vattn.h"

VATTN_NAMESPACE_BEGIN

class VAttnException : public std::exception {
public:
    explicit VAttnException(const std::string &msg);

    VAttnException(const std::string &msg,
                   const char *func_name,
                   const char *file,
                   int line);

    /// from std::exception
    const char *what() const noexcept override;

    std::string msg;
};

VATTN_NAMESPACE_END