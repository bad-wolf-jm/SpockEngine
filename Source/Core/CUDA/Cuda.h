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
#    define SE_CUDA_DEVICE_FUNCTION_DEF __device__
#    define SE_CUDA_INLINE __forceinline__
#    define CUDA_KERNEL_DEFINITION __global__
#else
#    define SE_CUDA_INLINE
#    define SE_CUDA_HOST_DEVICE_FUNCTION_DEF
#    define SE_CUDA_DEVICE_FUNCTION_DEF
#    define CUDA_KERNEL_DEFINITION
#endif

#define RETURN_UNLESS( condition )   \
    do                               \
    {                                \
        if( !( condition ) ) return; \
    } while( 0 )

namespace SE::Cuda
{
    using namespace SE::Core;

    using RawPointer     = CUdeviceptr;
    using Array          = cudaArray_t;
    using MipmappedArray = cudaMipmappedArray_t;
    using ExternalMemory = cudaExternalMemory_t;
    using TextureObject  = cudaTextureObject_t;

    void SyncDevice();

    void Malloc( void **aDestination, size_t aSize );
    void Free( void **aDestination );
    void MemCopyHostToDevice( void *aDestination, void *aSource, size_t aSize );
    void MemCopyDeviceToHost( void *aDestination, void *aSource, size_t aSize );

    void MallocArray( Array *aDestination, eColorFormat aFormat, size_t aWidth, size_t aHeight );
    void FreeArray( Array *aDestination );
    void ArrayCopyHostToDevice( Array aDestination, size_t aWidthOffset, size_t aHeightOffset, void *aSource, size_t aSize );
    void ArrayCopyDeviceToHost( Array aDestination, void *aSource, size_t aWidthOffset, size_t aHeightOffset, size_t aSize );

    void ImportExternalMemory( ExternalMemory *aDestination, void *aExternalBuffer, size_t aSize );
    void DestroyExternalMemory( ExternalMemory *aDestination );

    void GetMappedMipmappedArray( MipmappedArray *aDestination, ExternalMemory aExternalMemoryHandle, eColorFormat aFormat,
                                  int32_t aWidth, int32_t aHeight );
    void GeMipmappedArrayLevel( Array *aDestination, MipmappedArray aMipMappedArray, uint32_t aLevel );
    void FreeMipmappedArray( MipmappedArray *aDestination );

    void CreateTextureObject( TextureObject *aDestination, Array aDataArray, sTextureSamplingInfo aSpec );
    void FreeTextureObject( TextureObject *aDestination );
} // namespace SE::Cuda
