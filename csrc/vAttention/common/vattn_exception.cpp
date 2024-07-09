#include "common/vattn_exception.h"

VATTN_NAMESPACE_BEGIN

VAttnException::VAttnException(const std::string &m) : msg(m) 
{ }

VAttnException::VAttnException(const std::string &m,
                               const char *func_name,
                               const char *file,
                               int line)
{
    int size = snprintf(nullptr,
                        0,
                        "Error in %s at %s:%d: %s",
                        func_name,
                        file,
                        line,
                        m.c_str());
    msg.resize(size + 1);
    snprintf(&msg[0],
             msg.size(),
             "Error in %s at %s:%d: %s",
             func_name,
             file,
             line,
             m.c_str());
}

const char *VAttnException::what() const noexcept 
{
    return msg.c_str();
}

VATTN_NAMESPACE_END
