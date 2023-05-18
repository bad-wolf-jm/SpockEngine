/// @file   TextureCubeMap.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "Conversion.h"
#include "TextureCubeMap.h"

using namespace SE::Core;
namespace SE::Cuda
{
    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, std::vector<uint8_t> aData )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData.data() ), aData.size() );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, uint8_t *aData, size_t aSize )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData ), aSize );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, sImageData &aImageData )
        : mSpec( aSpec )
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;

        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, aImageData.mPixelData.data(), aImageData.mByteSize );
    }

    TextureCubeMap::TextureCubeMap( sTextureCreateInfo &aSpec, void *aExternalBuffer, size_t aImageMemorySize )
        : mSpec( aSpec )
        , mImageMemorySize{ aImageMemorySize }
    {
        ImportExternalMemory( &mExternalMemoryHandle, aExternalBuffer, aImageMemorySize );
        GetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle, mSpec.mFormat, mSpec.mWidth, mSpec.mHeight );
        GeMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 );
    }

    TextureCubeMap::~TextureCubeMap()
    {
        FreeArray( &mInternalCudaArray );
        FreeMipmappedArray( &mInternalCudaMipmappedArray );
        DestroyExternalMemory( &mExternalMemoryHandle );
    }

    TextureSamplerCubeMap::TextureSamplerCubeMap( Ref<TextureCubeMap> &aTexture, const sTextureSamplingInfo &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void TextureSamplerCubeMap::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda