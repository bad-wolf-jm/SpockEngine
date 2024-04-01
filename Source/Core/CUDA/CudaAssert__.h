/// @file   CudaAssert.h
///
/// @brief  Helper macros for Cuda
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "Core/Logging.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fmt/core.h>
#include <stdexcept>

#ifndef CUDA_ASSERT
#    define CUDA_ASSERT( err ) __CUDA_ASSERT( (cudaError_t)err, __FILE__, __LINE__ )

inline void __CUDA_ASSERT( cudaError_t err, const char *file, const int line )
{
    if( CUDA_SUCCESS == err )
        return;

    const char *errorStr = cudaGetErrorString( err );
    SE::Logging::Error( "CUDA_ASSERT() API error = {} \"{}\" from file <{}>, line {}.\n", err, errorStr, file, line );
    throw std::runtime_error( "CUDA_ASSERT()" );
}

#endif
