#include "utils/timer.h"

VATTN_NAMESPACE_BEGIN

Timer::Timer() 
{
    start_ = std::chrono::steady_clock::now();
}

void Timer::reset()
{
    start_ = std::chrono::steady_clock::now();
}
    
float Timer::elapsed_seconds() const 
{
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> duration = end - start_;
    return duration.count();
}

float Timer::elapsed_milliseconds() const 
{
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start_;
    return duration.count();
}
    
float Timer::elapsed_microseconds() const
{
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::micro> duration = end - start_;
    return duration.count();
}
    
int64_t Timer::get_millisecs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

VATTN_NAMESPACE_END