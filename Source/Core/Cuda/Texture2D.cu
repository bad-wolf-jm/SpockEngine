/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include "Texture2D.h"

using namespace SE::Core;
namespace SE::Cuda
{
    /// @brief Convert our internal color format into a CUDA channel description
    static cudaChannelFormatDesc ToCudaChannelDesc( eColorFormat aColorFormat )
    {
        switch( aColorFormat )
        {
        case eColorFormat::R32_FLOAT:
            return cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RG32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RGB32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 32, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RGBA32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
        case eColorFormat::R8_UNORM:
            return cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RG8_UNORM:
            return cudaCreateChannelDesc( 8, 8, 0, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RGB8_UNORM:
            return cudaCreateChannelDesc( 8, 8, 8, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RGBA8_UNORM:
        default:
            return cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned );
        }
    }

    /// @brief Convert our wrapping descriptor into a CUDA wrapping descriptor
    static cudaTextureAddressMode ToCudaAddressMode( eSamplerWrapping aAddressMode )
    {
        switch( aAddressMode )
        {
        case eSamplerWrapping::REPEAT:
            return cudaAddressModeWrap;
        case eSamplerWrapping::MIRRORED_REPEAT:
            return cudaAddressModeMirror;
        case eSamplerWrapping::CLAMP_TO_EDGE:
            return cudaAddressModeClamp;
        case eSamplerWrapping::CLAMP_TO_BORDER:
        case eSamplerWrapping::MIRROR_CLAMP_TO_BORDER:
        default:
            return cudaAddressModeBorder;
        }
    }

    /// @brief Convert our filtering descriptor into a CUDA filtering descriptor
    static cudaTextureFilterMode ToCudaFilterMode( eSamplerFilter aFilterMode )
    {
        switch( aFilterMode )
        {
        case eSamplerFilter::NEAREST:
            return cudaFilterModePoint;
        case eSamplerFilter::LINEAR:
        default:
            return cudaFilterModeLinear;
        }
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, std::vector<uint8_t> aData )
        : mSpec( aSpec )
    {
        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
            static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT( cudaMemcpyToArray(
            mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData.data() ), aData.size(), cudaMemcpyHostToDevice ) );
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, uint8_t *aData, size_t aSize )
        : mSpec( aSpec )
    {
        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
            static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT( cudaMemcpyToArray( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData ), aSize, cudaMemcpyHostToDevice ) );
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, sImageData &aImageData )
        : mSpec( aSpec )
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;

        cudaChannelFormatDesc lTextureFormat = ToCudaChannelDesc( mSpec.mFormat );

        CUDA_ASSERT( cudaMallocArray( &mInternalCudaArray, &lTextureFormat, static_cast<size_t>( mSpec.mWidth ),
            static_cast<size_t>( mSpec.mHeight ), cudaArrayDefault ) );
        CUDA_ASSERT(
            cudaMemcpyToArray( mInternalCudaArray, 0, 0, aImageData.mPixelData, aImageData.mByteSize, cudaMemcpyHostToDevice ) );
    }

    Texture2D::~Texture2D()
    {
        if( ( nullptr != ( (void *)mInternalCudaArray ) ) ) CUDA_ASSERT( cudaFreeArray( mInternalCudaArray ) );
    }

    TextureSampler2D::TextureSampler2D( Ref<Texture2D> &aTexture, const sTextureSamplingInfo &aSamplingSpec )
        : mTexture{ aTexture }
        , mSamplingSpec{ aSamplingSpec }
    {
        cudaResourceDesc lResourceDescription{};
        memset( &lResourceDescription, 0, sizeof( cudaResourceDesc ) );

        lResourceDescription.resType         = cudaResourceTypeArray;
        lResourceDescription.res.array.array = mTexture->mInternalCudaArray;

        cudaTextureDesc lTextureDescription{};
        memset( &lTextureDescription, 0, sizeof( cudaTextureDesc ) );

        lTextureDescription.readMode = cudaReadModeElementType;
        if( mTexture->mSpec.mNormalizedValues ) lTextureDescription.readMode = cudaReadModeNormalizedFloat;
        lTextureDescription.borderColor[0] = mSamplingSpec.mBorderColor[0];
        lTextureDescription.borderColor[1] = mSamplingSpec.mBorderColor[1];
        lTextureDescription.borderColor[2] = mSamplingSpec.mBorderColor[2];
        lTextureDescription.borderColor[3] = mSamplingSpec.mBorderColor[3];

        lTextureDescription.addressMode[0] = ToCudaAddressMode( mSamplingSpec.mWrapping );
        lTextureDescription.addressMode[1] = ToCudaAddressMode( mSamplingSpec.mWrapping );
        lTextureDescription.addressMode[2] = ToCudaAddressMode( mSamplingSpec.mWrapping );

        lTextureDescription.filterMode = ToCudaFilterMode( mSamplingSpec.mMagnification );

        lTextureDescription.normalizedCoords = 0;
        if( mTexture->mSpec.mNormalizedCoordinates ) lTextureDescription.normalizedCoords = 1;

        lTextureDescription.mipmapFilterMode    = cudaFilterModePoint;
        lTextureDescription.mipmapLevelBias     = 0.0f;
        lTextureDescription.minMipmapLevelClamp = 0.0f;
        lTextureDescription.maxMipmapLevelClamp = 1.0f;

        mDeviceData.mScaling = math::vec2{ aSamplingSpec.mScaling[0], aSamplingSpec.mScaling[1] };
        mDeviceData.mOffset  = math::vec2{ aSamplingSpec.mOffset[0], aSamplingSpec.mOffset[1] };
        CUDA_ASSERT( cudaCreateTextureObject( &( mDeviceData.mTextureObject ), &lResourceDescription, &lTextureDescription, NULL ) );
    }

} // namespace SE::Cuda