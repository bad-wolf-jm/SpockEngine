#include "Cuda.h"

#define CUDA_INTEROP
// #define CUDA_INTEROP

#ifdef CUDA_INTEROP
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#endif

#include "Texture/Conversion.h"

namespace SE::Cuda
{

#ifndef CUDA_INTEROP
#    ifndef CUDA_ASSERT
#        define CUDA_ASSERT( err ) \
            do                     \
            {                      \
            } while( 0 )
#    endif
#else
#    ifndef CUDA_ASSERT
#        define CUDA_ASSERT( err ) __CUDA_ASSERT( (cudaError_t)err, __FILE__, __LINE__ )

    inline void __CUDA_ASSERT( cudaError_t aErr, const char *aFile, const int aLine )
    {
        if( CUDA_SUCCESS == aErr ) return;

        const char *errorStr = cudaGetErrorString( aErr );
        SE::Logging::Error( "CUDA_ASSERT() API error = {} \"{}\" from file <{}>, line {}.\n", aErr, errorStr, aFile, aLine );
        throw std::runtime_error( "CUDA_ASSERT()" );
    }

#        define CUDA_SYNC_CHECK() CUDA_ASSERT( cudaDeviceSynchronize() )
#    endif
#endif

    void Malloc( void **aDestination, size_t aSize ) { CUDA_ASSERT( cudaMalloc( aDestination, aSize ) ); }

    void Free( void **aDestination )
    {
        if( nullptr != aDestination ) CUDA_ASSERT( cudaFree( *aDestination ) );

        *aDestination = nullptr;
    }

    void MemCopyHostToDevice( void *aDestination, void *aSource, size_t aSize )
    {
        CUDA_ASSERT( cudaMemcpy( aDestination, aSource, aSize, cudaMemcpyHostToDevice ) );
    }

    void MemCopyDeviceToHost( void *aDestination, void *aSource, size_t aSize )
    {
        CUDA_ASSERT( cudaMemcpy( aDestination, aSource, aSize, cudaMemcpyDeviceToHost ) );
    }

    void MallocArray( void **aDestination, eColorFormat aFormat, size_t aWidth, size_t aHeight )
    {
#ifdef CUDA_INTEROP
        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( aFormat );
        CUDA_ASSERT(
            cudaMallocArray( reinterpret_cast<cudaArray_t *>( aDestination ), &lTextureFormat, aWidth, aHeight, cudaArrayDefault ) );
#endif
    }

    void FreeArray( void **aDestination )
    {
        if( nullptr != aDestination ) CUDA_ASSERT( cudaFreeArray( reinterpret_cast<cudaArray_t>( *aDestination ) ) );

        *aDestination = nullptr;
    }

    void ArrayCopyHostToDevice( void *aDestination, size_t aWidthOffset, size_t aHeightOffset, void *aSource, size_t aSize )
    {
        CUDA_ASSERT( cudaMemcpyToArray( reinterpret_cast<cudaArray_t>( aDestination ), aWidthOffset, aHeightOffset, aSource, aSize,
                                        cudaMemcpyHostToDevice ) );
    }

    void ArrayCopyDeviceToHost( void *aDestination, void *aSource, size_t aWidthOffset, size_t aHeightOffset, size_t aSize )
    {
        CUDA_ASSERT( cudaMemcpyFromArray( reinterpret_cast<cudaArray_t>( aDestination ),
                                          reinterpret_cast<cudaArray_const_t>( aSource ), aWidthOffset, aHeightOffset, aSize,
                                          cudaMemcpyDeviceToHost ) );
    }

    void ImportExternalMemory( void **aDestination, void *aExternalBuffer, size_t aSize )
    {
#ifdef CUDA_INTEROP
        cudaExternalMemoryHandleDesc lCudaExternalMemoryHandleDesc{};
        lCudaExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        lCudaExternalMemoryHandleDesc.size                = aSize;
        lCudaExternalMemoryHandleDesc.flags               = 0;
        lCudaExternalMemoryHandleDesc.handle.win32.handle = aExternalBuffer;

        CUDA_ASSERT(
            cudaImportExternalMemory( reinterpret_cast<cudaExternalMemory_t *>( aDestination ), &lCudaExternalMemoryHandleDesc ) );
#endif
    }

    void DestroyExternalMemory( void **aDestination )
    {
        if( nullptr != aDestination )
            CUDA_ASSERT( cudaDestroyExternalMemory( reinterpret_cast<cudaExternalMemory_t>( *aDestination ) ) );

        *aDestination = nullptr;
    }

    void GetMappedMipmappedArray( void **aDestination, void *aExternalMemoryHandle, eColorFormat aFormat, int32_t aWidth,
                                  int32_t aHeight, size_t aSize )
    {
#ifdef CUDA_INTEROP
        cudaExternalMemoryMipmappedArrayDesc lExternalMemoryMipmappedArrayDesc{};
        lExternalMemoryMipmappedArrayDesc.formatDesc    = ToCudaChannelDesc( aFormat );
        lExternalMemoryMipmappedArrayDesc.extent.width  = aWidth;
        lExternalMemoryMipmappedArrayDesc.extent.height = aHeight;
        lExternalMemoryMipmappedArrayDesc.extent.depth  = 0;
        lExternalMemoryMipmappedArrayDesc.numLevels     = 1;
        lExternalMemoryMipmappedArrayDesc.flags         = 0;

        CUDA_ASSERT( cudaExternalMemoryGetMappedMipmappedArray( reinterpret_cast<cudaMipmappedArray_t *>( aDestination ),
                                                                reinterpret_cast<cudaExternalMemory_t>( aExternalMemoryHandle ),
                                                                &lExternalMemoryMipmappedArrayDesc ) );
#endif
    }

    void FreeMipmappedArray( void **aDestination )
    {
        if( nullptr != aDestination ) CUDA_ASSERT( cudaFreeMipmappedArray( reinterpret_cast<cudaMipmappedArray_t>( *aDestination ) ) );

        *aDestination = nullptr;
    }

    void GeMipmappedArrayLevel( void **aDestination, void *aMipMappedArray, uint32_t aLevel )
    {
        CUDA_ASSERT( cudaGetMipmappedArrayLevel( reinterpret_cast<cudaArray_t *>( aDestination ),
                                                 reinterpret_cast<cudaMipmappedArray_t>( aMipMappedArray ), aLevel ) );
    }

    void CreateTextureObject( void **aDestination, void *aDataArray, sTextureSamplingInfo aSpec )
    {
#ifdef CUDA_INTEROP
        cudaResourceDesc lResourceDescription{};
        memset( &lResourceDescription, 0, sizeof( cudaResourceDesc ) );

        lResourceDescription.resType         = cudaResourceTypeArray;
        lResourceDescription.res.array.array = reinterpret_cast<cudaArray_t>( aDataArray );

        cudaTextureDesc lTextureDescription{};
        memset( &lTextureDescription, 0, sizeof( cudaTextureDesc ) );

        lTextureDescription.readMode = cudaReadModeElementType;
        if( aSpec.mNormalizedValues ) lTextureDescription.readMode = cudaReadModeNormalizedFloat;

        lTextureDescription.borderColor[0] = aSpec.mBorderColor[0];
        lTextureDescription.borderColor[1] = aSpec.mBorderColor[1];
        lTextureDescription.borderColor[2] = aSpec.mBorderColor[2];
        lTextureDescription.borderColor[3] = aSpec.mBorderColor[3];

        lTextureDescription.addressMode[0] = ToCudaAddressMode( aSpec.mWrapping );
        lTextureDescription.addressMode[1] = ToCudaAddressMode( aSpec.mWrapping );
        lTextureDescription.addressMode[2] = ToCudaAddressMode( aSpec.mWrapping );

        lTextureDescription.filterMode = ToCudaFilterMode( aSpec.mFilter );

        lTextureDescription.normalizedCoords = 0;
        if( aSpec.mNormalizedCoordinates ) lTextureDescription.normalizedCoords = 1;

        lTextureDescription.mipmapFilterMode    = cudaFilterModePoint;
        lTextureDescription.mipmapLevelBias     = 0.0f;
        lTextureDescription.minMipmapLevelClamp = 0.0f;
        lTextureDescription.maxMipmapLevelClamp = 1.0f;

        CUDA_ASSERT( cudaCreateTextureObject( reinterpret_cast<cudaTextureObject_t *>( aDestination ), &lResourceDescription,
                                              &lTextureDescription, NULL ) );
#endif
    }

    void FreeTextureObject( void **aDestination )
    {
        if( nullptr != aDestination )
            CUDA_ASSERT( cudaDestroyTextureObject( reinterpret_cast<cudaTextureObject_t>( *aDestination ) ) );

        *aDestination = nullptr;

        // CUDA_ASSERT( cudaMalloc( reinterpret_cast<void **>( aDestination ), aSize * sizeof( _Ty ) ) );
    }

} // namespace SE::Cuda