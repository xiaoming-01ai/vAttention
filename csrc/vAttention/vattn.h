#pragma once

#include <stdio.h>
#include <memory>
#include <cstdint>
#include <cassert>
#include <limits>
#include <string.h>
#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define VATTN_NAMESPACE_BEGIN namespace vAttention { 
#define VATTN_NAMESPACE_END } 
#define VATTN_NAMESPACE_USE using namespace vAttention;

VATTN_NAMESPACE_BEGIN

// uint64_t create_attention();

// bool attention();

// void release_attention(uint64_t attn_id);

VATTN_NAMESPACE_END