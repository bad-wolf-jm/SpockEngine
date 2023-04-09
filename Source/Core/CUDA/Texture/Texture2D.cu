/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "Conversion.h"
#include "Texture2D.h"

using namespace SE::Core;
namespace SE::Cuda
{
    Texture2D::Texture2D( sTextureCreateInfo &aSpec, std::vector<uint8_t> aData )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData.data() ), aData.size() );
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, uint8_t *aData, size_t aSize )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData ), aSize );
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, sImageData &aImageData )
        : mSpec( aSpec )
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;

        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, aImageData.mPixelData.data(), aImageData.mByteSize );
    }

    Texture2D::Texture2D( sTextureCreateInfo &aSpec, void *aExternalBuffer, size_t aImageMemorySize )
        : mSpec( aSpec )
        , mImageMemorySize{ aImageMemorySize }
    {
        ImportExternalMemory( &mExternalMemoryHandle, aExternalBuffer, aImageMemorySize );
        GetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle, mSpec.mFormat, mSpec.mWidth, mSpec.mHeight );
        GeMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 );
    }

    Texture2D::~Texture2D()
    {
        FreeArray( &mInternalCudaArray );
        FreeMipmappedArray( &mInternalCudaMipmappedArray );
        DestroyExternalMemory( &mExternalMemoryHandle );
    }

    TextureSampler2D::TextureSampler2D( Ref<Texture2D> &aTexture, const sTextureSamplingInfo &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void TextureSampler2D::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda