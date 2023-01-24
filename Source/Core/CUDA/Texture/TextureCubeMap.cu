/// @file   TextureCubeMap.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "Conversion.h"
#include "TextureCubeMap.h"

using namespace SE::Core;
namespace SE::Cuda
{
    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, std::vector<uint8_t> aData )
        : mSpec( aSpec )
    {
        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
                                      static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT( cudaMemcpyToArray( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData.data() ), aData.size(),
                                        cudaMemcpyHostToDevice ) );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, uint8_t *aData, size_t aSize )
        : mSpec( aSpec )
    {
        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
                                      static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT( cudaMemcpyToArray( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData ), aSize, cudaMemcpyHostToDevice ) );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, sImageData &aImageData )
        : mSpec( aSpec )
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;

        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
                                      static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT( cudaMemcpyToArray( mInternalCudaArray, 0, 0, aImageData.mPixelData.data(), aImageData.mByteSize,
                                        cudaMemcpyHostToDevice ) );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, void *aExternalBuffer, size_t aImageMemorySize )
        : mSpec( aSpec )
        , mImageMemorySize{ aImageMemorySize }
    {
        cudaExternalMemoryHandleDesc lCudaExternalMemoryHandleDesc{};
        lCudaExternalMemoryHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        lCudaExternalMemoryHandleDesc.size                = mImageMemorySize;
        lCudaExternalMemoryHandleDesc.flags               = 0;
        lCudaExternalMemoryHandleDesc.handle.win32.handle = aExternalBuffer;

        CUDA_ASSERT( cudaImportExternalMemory( &mExternalMemoryHandle, &lCudaExternalMemoryHandleDesc ) );

        cudaExternalMemoryMipmappedArrayDesc lExternalMemoryMipmappedArrayDesc{};
        lExternalMemoryMipmappedArrayDesc.formatDesc = ToCudaChannelDesc( mSpec.mFormat );
        lExternalMemoryMipmappedArrayDesc.extent     = make_cudaExtent( mSpec.mWidth, mSpec.mHeight, 0 );
        lExternalMemoryMipmappedArrayDesc.numLevels  = 1;
        lExternalMemoryMipmappedArrayDesc.flags      = 0;

        CUDA_ASSERT( cudaExternalMemoryGetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle,
                                                                &lExternalMemoryMipmappedArrayDesc ) );
        CUDA_ASSERT( cudaGetMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 ) );
    }

    TextureCubeMap::~TextureCubeMap()
    {
        if( ( nullptr != ( (void *)mInternalCudaArray ) ) ) CUDA_ASSERT( cudaFreeArray( mInternalCudaArray ) );
        mInternalCudaArray = nullptr;

        if( ( nullptr != ( (void *)mInternalCudaMipmappedArray ) ) )
            CUDA_ASSERT( cudaFreeMipmappedArray( mInternalCudaMipmappedArray ) );
        mInternalCudaMipmappedArray = nullptr;

        if( mExternalMemoryHandle ) CUDA_ASSERT( cudaDestroyExternalMemory( mExternalMemoryHandle ) );
        mExternalMemoryHandle = nullptr;
    }

    TextureSamplerCubeMap::TextureSamplerCubeMap( Ref<TextureCubeMap> &aTexture, const sTextureSamplingInfo &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void TextureSamplerCubeMap::InitializeTextureSampler()
    {
        cudaResourceDesc lResourceDescription{};
        memset( &lResourceDescription, 0, sizeof( cudaResourceDesc ) );

        lResourceDescription.resType         = cudaResourceTypeArray;
        lResourceDescription.res.array.array = mTexture->mInternalCudaArray;

        cudaTextureDesc lTextureDescription{};
        memset( &lTextureDescription, 0, sizeof( cudaTextureDesc ) );

        lTextureDescription.readMode = cudaReadModeElementType;
        if( mSpec.mNormalizedValues ) lTextureDescription.readMode = cudaReadModeNormalizedFloat;

        lTextureDescription.borderColor[0] = mSpec.mBorderColor[0];
        lTextureDescription.borderColor[1] = mSpec.mBorderColor[1];
        lTextureDescription.borderColor[2] = mSpec.mBorderColor[2];
        lTextureDescription.borderColor[3] = mSpec.mBorderColor[3];

        lTextureDescription.addressMode[0] = ToCudaAddressMode( mSpec.mWrapping );
        lTextureDescription.addressMode[1] = ToCudaAddressMode( mSpec.mWrapping );
        lTextureDescription.addressMode[2] = ToCudaAddressMode( mSpec.mWrapping );

        lTextureDescription.filterMode = ToCudaFilterMode( mSpec.mFilter );

        lTextureDescription.normalizedCoords = 0;
        if( mSpec.mNormalizedCoordinates ) lTextureDescription.normalizedCoords = 1;

        lTextureDescription.mipmapFilterMode    = cudaFilterModePoint;
        lTextureDescription.mipmapLevelBias     = 0.0f;
        lTextureDescription.minMipmapLevelClamp = 0.0f;
        lTextureDescription.maxMipmapLevelClamp = 1.0f;

        mDeviceData.mScaling = math::vec2{ mSpec.mScaling[0], mSpec.mScaling[1] };
        mDeviceData.mOffset  = math::vec2{ mSpec.mOffset[0], mSpec.mOffset[1] };
        CUDA_ASSERT( cudaCreateTextureObject( &( mDeviceData.mTextureObject ), &lResourceDescription, &lTextureDescription, NULL ) );
    }

} // namespace SE::Cuda