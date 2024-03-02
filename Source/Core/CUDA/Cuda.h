#pragma once

#include <fmt/core.h>
#include <stdexcept>
#include <type_traits>

// #define CUDA_INTEROP
// #define CUDA_INTEROP_TYPE false

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Core/Logging.h"
#include "Texture/ColorFormat.h"
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

    void Malloc( void **aDestination, size_t aSize );
    void Free( void **aDestination );
    void MemCopyHostToDevice( void *aDestination, void *aSource, size_t aSize );
    void MemCopyDeviceToHost( void *aDestination, void *aSource, size_t aSize );

    void MallocArray( array_t *aDestination, color_format_t aFormat, size_t aWidth, size_t aHeight );
    void FreeArray( array_t *aDestination );
    void ArrayCopyHostToDevice( array_t aDestination, size_t aWidthOffset, size_t aHeightOffset, void *aSource, size_t aSize );
    void ArrayCopyDeviceToHost( array_t aDestination, void *aSource, size_t aWidthOffset, size_t aHeightOffset, size_t aSize );

    void ImportExternalMemory( external_memory_t *aDestination, void *aExternalBuffer, size_t aSize );
    void DestroyExternalMemory( external_memory_t *aDestination );

    void GetMappedMipmappedArray( mipmapped_array_t *aDestination, external_memory_t aExternalMemoryHandle, color_format_t aFormat,
                                  int32_t aWidth, int32_t aHeight );
    void GeMipmappedArrayLevel( array_t *aDestination, mipmapped_array_t aMipMappedArray, uint32_t aLevel );
    void FreeMipmappedArray( mipmapped_array_t *aDestination );

    void CreateTextureObject( texture_object_t *aDestination, array_t aDataArray, texture_sampling_info_t aSpec );
    void FreeTextureObject( texture_object_t *aDestination );
} // namespace SE::Cuda
