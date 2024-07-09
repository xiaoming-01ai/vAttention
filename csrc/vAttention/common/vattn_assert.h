#pragma once

#include "vattn.h"
#include "common/vattn_exception.h"

#include <cstdio>
#include <cstdlib>
#include <string>

///
/// Assertions
///

#define VATTN_ASSERT(X) \
    do { \
        if (!(X)) { \
            fprintf(stderr, \
                    "VATTN assertion '%s' failed in %s " \
                    "at %s:%d\n", \
                    #X, \
                    __PRETTY_FUNCTION__, \
                    __FILE__, \
                    __LINE__); \
            abort(); \
        } \
    } while (false)

#define VATTN_ASSERT_MSG(X, MSG) \
    do { \
        if (!(X)) { \
            fprintf(stderr, \
                    "VATTN assertion '%s' failed in %s " \
                    "at %s:%d; details: " MSG "\n", \
                    #X, \
                    __PRETTY_FUNCTION__, \
                    __FILE__, \
                    __LINE__); \
            abort(); \
        } \
    } while (false)

#define VATTN_ASSERT_FMT(X, FMT, ...) \
    do { \
        if (!(X)) { \
            fprintf(stderr, \
                    "VATTN assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n", \
                    #X, \
                    __PRETTY_FUNCTION__, \
                    __FILE__, \
                    __LINE__, \
                    __VA_ARGS__); \
            abort(); \
        } \
    } while (false) 

///
/// Exceptions for returning user errors
///

#define VATTN_THROW_MSG(MSG) \
    do { \
        throw vAttention::VAttnException(  \
                MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define VATTN_THROW_FMT(FMT, ...) \
    do { \
        std::string __s; \
        int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__); \
        __s.resize(__size + 1); \
        snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__); \
        throw vAttention::VAttnException( \
                __s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

///
/// Exceptions thrown upon a conditional failure
///

#define VATTN_THROW_IF_NOT(X) \
    do { \
        if (!(X)) { \
            VATTN_THROW_FMT("Error: '%s' failed", #X); \
        } \
    } while (false)

#define VATTN_THROW_IF_NOT_MSG(X, MSG) \
    do { \
        if (!(X)) { \
            VATTN_THROW_FMT("Error: '%s' failed: " MSG, #X); \
        } \
    } while (false)

#define VATTN_THROW_IF_NOT_FMT(X, FMT, ...) \
    do { \
        if (!(X)) { \
            VATTN_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }  \
    } while (false);
    