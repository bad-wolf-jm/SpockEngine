/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include "Conversion.h"
#include "Texture2D.h"

using namespace SE::Core;
namespace SE::Cuda
{
    texture2d_t::texture2d_t( texture_create_info_t &aSpec, vector_t<uint8_t> aData )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData.data() ), aData.size() );
    }

    texture2d_t::texture2d_t( texture_create_info_t &aSpec, uint8_t *aData, size_t aSize )
        : mSpec( aSpec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( aData ), aSize );
    }

    texture2d_t::texture2d_t( texture_create_info_t &aSpec, image_data_t &aImageData )
        : mSpec( aSpec )
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;

        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, aImageData.mPixelData.data(), aImageData.mByteSize );
    }

    texture2d_t::texture2d_t( texture_create_info_t &aSpec, void *aExternalBuffer, size_t aImageMemorySize )
        : mSpec( aSpec )
        , mImageMemorySize{ aImageMemorySize }
    {
        ImportExternalMemory( &mExternalMemoryHandle, aExternalBuffer, aImageMemorySize );
        GetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle, mSpec.mFormat, mSpec.mWidth, mSpec.mHeight );
        GeMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 );
    }

    texture2d_t::~texture2d_t()
    {
        FreeArray( &mInternalCudaArray );
        FreeMipmappedArray( &mInternalCudaMipmappedArray );
        DestroyExternalMemory( &mExternalMemoryHandle );
    }

    texture_sampler2d_t::texture_sampler2d_t( ref_t<texture2d_t> &aTexture, const texture_sampling_info_t &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void texture_sampler2d_t::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda