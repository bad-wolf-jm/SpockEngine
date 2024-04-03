#include "Cuda.h"

#include "Texture/Conversion.h"

namespace numlua::cuda
{
    void SyncDevice()
    {
        CUDA_ASSERT( cudaDeviceSynchronize() );
    }

    void Malloc( void **destination, size_t size )
    {
        CUDA_ASSERT( cudaMalloc( destination, size ) );
    }

    void Free( void **destination )
    {
        if( nullptr != destination )
            CUDA_ASSERT( cudaFree( *destination ) );

        *destination = nullptr;
    }

    void MemCopyHostToDevice( void *destination, void *source, size_t size )
    {
        CUDA_ASSERT( cudaMemcpy( destination, source, size, cudaMemcpyHostToDevice ) );
    }

    void MemCopyDeviceToHost( void *destination, void *source, size_t size )
    {
        CUDA_ASSERT( cudaMemcpy( destination, source, size, cudaMemcpyDeviceToHost ) );
    }

    void MallocArray( array_t *destination, color_format format, size_t width, size_t height )
    {
        cudaChannelFormatDesc textureFormat = ToCudaChannelDesc( format );
        CUDA_ASSERT( cudaMallocArray( destination, &textureFormat, width, height, cudaArrayDefault ) );
    }

    void FreeArray( array_t *destination )
    {
        if( nullptr != destination )
            CUDA_ASSERT( cudaFreeArray( *destination ) );

        *destination = nullptr;
    }

    void ArrayCopyHostToDevice( array_t destination, size_t widthOffset, size_t heightOffset, void *source, size_t size )
    {
        CUDA_ASSERT( cudaMemcpyToArray( destination, widthOffset, heightOffset, source, size, cudaMemcpyHostToDevice ) );
    }

    void ArrayCopyDeviceToHost( array_t destination, void *source, size_t widthOffset, size_t heightOffset, size_t size )
    {
        CUDA_ASSERT( cudaMemcpyFromArray( destination, reinterpret_cast<cudaArray_const_t>( source ), widthOffset, heightOffset,
                                          size, cudaMemcpyDeviceToHost ) );
    }

    void ImportExternalMemory( external_memory_t *destination, void *externalBuffer, size_t size )
    {
        cudaExternalMemoryHandleDesc cudaExternalMemoryHandleDesc{};
        cudaExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExternalMemoryHandleDesc.size                = size;
        cudaExternalMemoryHandleDesc.flags               = 0;
        cudaExternalMemoryHandleDesc.handle.win32.handle = externalBuffer;

        CUDA_ASSERT( cudaImportExternalMemory( destination, &cudaExternalMemoryHandleDesc ) );
    }

    void DestroyExternalMemory( external_memory_t *destination )
    {
        if( nullptr != *destination )
            CUDA_ASSERT( cudaDestroyExternalMemory( *destination ) );

        *destination = nullptr;
    }

    void GetMappedMipmappedArray( mipmapped_array_t *destination, external_memory_t externalMemoryHandle, color_format format,
                                  int32_t width, int32_t height )
    {
        cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc{};
        externalMemoryMipmappedArrayDesc.formatDesc    = ToCudaChannelDesc( format );
        externalMemoryMipmappedArrayDesc.extent.width  = width;
        externalMemoryMipmappedArrayDesc.extent.height = height;
        externalMemoryMipmappedArrayDesc.extent.depth  = 0;
        externalMemoryMipmappedArrayDesc.numLevels     = 1;
        externalMemoryMipmappedArrayDesc.flags         = 0;

        CUDA_ASSERT(
            cudaExternalMemoryGetMappedMipmappedArray( destination, externalMemoryHandle, &externalMemoryMipmappedArrayDesc ) );
    }

    void FreeMipmappedArray( mipmapped_array_t *destination )
    {
        if( nullptr != *destination )
            CUDA_ASSERT( cudaFreeMipmappedArray( reinterpret_cast<cudaMipmappedArray_t>( *destination ) ) );

        *destination = nullptr;
    }

    void GeMipmappedArrayLevel( array_t *destination, mipmapped_array_t mipMappedArray, uint32_t level )
    {
        CUDA_ASSERT( cudaGetMipmappedArrayLevel( destination, mipMappedArray, level ) );
    }

    void CreateTextureObject( texture_object_t *destination, array_t dataArray, texture_sampling_info_t spec )
    {
        cudaResourceDesc lResourceDescription{};
        memset( &lResourceDescription, 0, sizeof( cudaResourceDesc ) );

        lResourceDescription.resType         = cudaResourceTypeArray;
        lResourceDescription.res.array.array = reinterpret_cast<cudaArray_t>( dataArray );

        cudaTextureDesc lTextureDescription{};
        memset( &lTextureDescription, 0, sizeof( cudaTextureDesc ) );

        lTextureDescription.readMode = cudaReadModeElementType;
        if( spec.mNormalizedValues )
            lTextureDescription.readMode = cudaReadModeNormalizedFloat;

        lTextureDescription.borderColor[0] = spec.mBorderColor[0];
        lTextureDescription.borderColor[1] = spec.mBorderColor[1];
        lTextureDescription.borderColor[2] = spec.mBorderColor[2];
        lTextureDescription.borderColor[3] = spec.mBorderColor[3];

        lTextureDescription.addressMode[0] = ToCudaAddressMode( spec.mWrapping );
        lTextureDescription.addressMode[1] = ToCudaAddressMode( spec.mWrapping );
        lTextureDescription.addressMode[2] = ToCudaAddressMode( spec.mWrapping );

        lTextureDescription.filterMode = ToCudaFilterMode( spec.mFilter );

        lTextureDescription.normalizedCoords = 0;
        if( spec.mNormalizedCoordinates )
            lTextureDescription.normalizedCoords = 1;

        lTextureDescription.mipmapFilterMode    = cudaFilterModePoint;
        lTextureDescription.mipmapLevelBias     = 0.0f;
        lTextureDescription.minMipmapLevelClamp = 0.0f;
        lTextureDescription.maxMipmapLevelClamp = 1.0f;

        CUDA_ASSERT( cudaCreateTextureObject( destination, &lResourceDescription, &lTextureDescription, NULL ) );
    }

    void FreeTextureObject( texture_object_t *destination )
    {
        if( 0 != *destination )
            CUDA_ASSERT( cudaDestroyTextureObject( *destination ) );

        *destination = 0;
    }

} // namespace SE::Cuda
