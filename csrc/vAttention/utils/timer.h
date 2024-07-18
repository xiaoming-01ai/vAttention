#pragma once 

#include "vattn.h"
#include <chrono>

VATTN_NAMESPACE_BEGIN

class Timer {
public:
    /// Creates and starts a new timer
    Timer();

    /// Returns elapsed time in seconds
    float elapsed_seconds() const;
    
    /// Returns elapsed time in milliseconds
    float elapsed_milliseconds() const;
    
    /// Returns elapsed time in microseconds
    float elapsed_microseconds() const;

    void reset();

    static int64_t get_millisecs();

private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

VATTN_NAMESPACE_END