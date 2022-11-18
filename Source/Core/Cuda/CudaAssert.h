/// @file   CudaAssert.h
///
/// @brief  Helper macros for Cuda
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Core/Logging.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fmt/core.h>
#include <stdexcept>

#ifndef CUDA_ASSERT
#    define CUDA_ASSERT( err ) __CUDA_ASSERT( (cudaError_t)err, __FILE__, __LINE__ )

inline void __CUDA_ASSERT( cudaError_t aErr, const char *aFile, const int aLine )
{
    if( CUDA_SUCCESS == aErr ) return;

    const char *errorStr = cudaGetErrorString( aErr );
    SE::Logging::Error( "CUDA_ASSERT() API error = {} \"{}\" from file <{}>, line {}.\n", aErr, errorStr, aFile, aLine );
    throw std::runtime_error( "CUDA_ASSERT()" );
}

#    define CUDA_SYNC_CHECK() CUDA_ASSERT( cudaDeviceSynchronize() )
#endif

#ifdef __CUDACC__
#    define LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF __device__ __host__
#    define LTSE_CUDA_DEVICE_FUNCTION_DEF __device__
#    define LTSE_CUDA_INLINE __forceinline__
#    define CUDA_KERNEL_DEFINITION __global__
#else
#    define LTSE_CUDA_INLINE
#    define LTSE_CUDA_HOST_DEVICE_FUNCTION_DEF
#    define LTSE_CUDA_DEVICE_FUNCTION_DEF
#    define CUDA_KERNEL_DEFINITION
#endif

#define RETURN_UNLESS( condition )   \
    do                               \
    {                                \
        if( !( condition ) ) return; \
    } while( 0 )
