/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include "Conversion.h"
#include "Texture2D.h"

using namespace numlua::core;
namespace numlua::cuda
{
    texture2d_t::texture2d_t( texture_create_info_t &spec, vector_t<uint8_t> data )
        : mSpec( spec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( data.data() ), data.size() );
    }

    texture2d_t::texture2d_t( texture_create_info_t &spec, uint8_t *data, size_t size )
        : mSpec( spec )
    {
        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, reinterpret_cast<void *>( data ), size );
    }

    texture2d_t::texture2d_t( texture_create_info_t &spec, image_data_t &imageData )
        : mSpec( spec )
    {
        mSpec.mFormat = imageData.mFormat;
        mSpec.mWidth  = imageData.mWidth;
        mSpec.mHeight = imageData.mHeight;

        MallocArray( &mInternalCudaArray, mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ) );
        ArrayCopyHostToDevice( mInternalCudaArray, 0, 0, imageData.mPixelData.data(), imageData.mByteSize );
    }

    texture2d_t::texture2d_t( texture_create_info_t &spec, void *externalBuffer, size_t imageMemorySize )
        : mSpec( spec )
        , mImageMemorySize{ imageMemorySize }
    {
        ImportExternalMemory( &mExternalMemoryHandle, externalBuffer, imageMemorySize );
        GetMappedMipmappedArray( &mInternalCudaMipmappedArray, mExternalMemoryHandle, mSpec.mFormat, mSpec.mWidth, mSpec.mHeight );
        GeMipmappedArrayLevel( &mInternalCudaArray, mInternalCudaMipmappedArray, 0 );
    }

    texture2d_t::~texture2d_t()
    {
        FreeArray( &mInternalCudaArray );
        FreeMipmappedArray( &mInternalCudaMipmappedArray );
        DestroyExternalMemory( &mExternalMemoryHandle );
    }

    texture_sampler2d_t::texture_sampler2d_t( ref_t<texture2d_t> &texture, const texture_sampling_info_t &samplingSpec )
        : mTexture{ texture }
        , mSpec{ samplingSpec }
    {
        InitializeTextureSampler();
    }

    void texture_sampler2d_t::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda