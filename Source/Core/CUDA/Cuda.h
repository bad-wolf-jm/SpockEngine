#pragma once

#include <fmt/core.h>
#include <stdexcept>
#include <type_traits>

// #define CUDA_INTEROP
// #define CUDA_INTEROP_TYPE false

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Core/Logging.h"
#include "Texture/TextureTypes.h"

#ifdef __CUDACC__
#    define SE_CUDA_HOST_DEVICE_FUNCTION_DEF __device__ __host__
#    define SE_CUDA_DEVICE_FUNCTION_DEF      __device__
#    define SE_CUDA_INLINE                   __forceinline__
#    define CUDA_KERNEL_DEFINITION           __global__
#else
#    define SE_CUDA_INLINE
#    define SE_CUDA_HOST_DEVICE_FUNCTION_DEF
#    define SE_CUDA_DEVICE_FUNCTION_DEF
#    define CUDA_KERNEL_DEFINITION
#endif

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

#define RETURN_UNLESS( condition ) \
    do                             \
    {                              \
        if( !( condition ) )       \
            return;                \
    } while( 0 )

namespace SE::Cuda
{
    using namespace SE::Core;

    using raw_pointer_t     = CUdeviceptr;
    using array_t           = cudaArray_t;
    using mipmapped_array_t = cudaMipmappedArray_t;
    using external_memory_t = cudaExternalMemory_t;
    using texture_object_t  = cudaTextureObject_t;

    void SyncDevice();

    void Malloc( void **destination, size_t size );
    void Free( void **destination );
    void MemCopyHostToDevice( void *destination, void *source, size_t size );
    void MemCopyDeviceToHost( void *destination, void *source, size_t size );

    void MallocArray( array_t *destination, color_format format, size_t width, size_t height );
    void FreeArray( array_t *destination );
    void ArrayCopyHostToDevice( array_t destination, size_t widthOffset, size_t heightOffset, void *source, size_t size );
    void ArrayCopyDeviceToHost( array_t destination, void *source, size_t widthOffset, size_t heightOffset, size_t size );

    void ImportExternalMemory( external_memory_t *destination, void *externalBuffer, size_t size );
    void DestroyExternalMemory( external_memory_t *destination );

    void GetMappedMipmappedArray( mipmapped_array_t *destination, external_memory_t externalMemoryHandle, color_format format,
                                  int32_t width, int32_t height );
    void GeMipmappedArrayLevel( array_t *destination, mipmapped_array_t mipMappedArray, uint32_t aLevel );
    void FreeMipmappedArray( mipmapped_array_t *destination );

    void CreateTextureObject( texture_object_t *destination, array_t dataArray, texture_sampling_info_t spec );
    void FreeTextureObject( texture_object_t *destination );
} // namespace SE::Cuda
