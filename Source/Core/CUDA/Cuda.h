#pragma once

#include <fmt/core.h>
#include <stdexcept>

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

    void Malloc( void **aDestination, size_t aSize );
    void Free( void **aDestination );
    void MemCopyHostToDevice( void *aDestination, void *aSource, size_t aSize );
    void MemCopyDeviceToHost( void *aDestination, void *aSource, size_t aSize );

    void MallocArray( void **aDestination, eColorFormat aFormat, size_t aWidth, size_t aHeight );
    void FreeArray( void **aDestination );
    void ArrayCopyHostToDevice( void *aDestination, size_t aWidthOffset, size_t aHeightOffset, void *aSource, size_t aSize );
    void ArrayCopyDeviceToHost( void *aDestination, void *aSource, size_t aWidthOffset, size_t aHeightOffset, size_t aSize );

    void ImportExternalMemory( void **aDestination, void *aExternalBuffer, size_t aSize );
    void DestroyExternalMemory( void **aDestination, size_t aSize );

    void GetMappedMipmappedArray( void **aDestination, void *aExternalMemoryHandle, eColorFormat aFormat, int32_t aWidth,
                                  int32_t aHeight, size_t aSize );
    void GeMipmappedArrayLevel( void **aDestination, void *aMipMappedArray, uint32_t aLevel );
    void FreeMipmappedArray( void **aDestination );

    void CreateTextureObject( void **aDestination, void *aDataArray, sTextureSamplingInfo aSpec );
    void FreeTextureObject( void **aDestination );
} // namespace SE::Cuda
