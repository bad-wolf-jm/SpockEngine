/// @file   TextureData.cpp
///
/// @brief  Implementation file for texture data
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include <set>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

// #include "Core/Core.h"
// #include "Core/Memory.h"

#include "TextureData.h"

#include "Core/Logging.h"

#include <filesystem>
#include <fstream>
#include <Core/Textures/generate_mipmaps.hpp>

namespace numlua::core
{
    static const format ToGliType( const color_format &aFormat )
    {
        switch( aFormat )
        {
        case color_format::R32_FLOAT: return FORMAT_R32_SFLOAT_PACK32;
        case color_format::RG32_FLOAT: return FORMAT_RG32_SFLOAT_PACK32;
        case color_format::RGB32_FLOAT: return FORMAT_RGB32_SFLOAT_PACK32;
        case color_format::RGBA32_FLOAT: return FORMAT_RGBA32_SFLOAT_PACK32;
        case color_format::R16_FLOAT: return FORMAT_R16_SFLOAT_PACK16;
        case color_format::RG16_FLOAT: return FORMAT_RG16_SFLOAT_PACK16;
        case color_format::RGB16_FLOAT: return FORMAT_RGB16_SFLOAT_PACK16;
        case color_format::RGBA16_FLOAT: return FORMAT_RGBA16_SFLOAT_PACK16;
        case color_format::R8_UNORM: return FORMAT_R8_UNORM_PACK8;
        case color_format::RG8_UNORM: return FORMAT_RG8_UNORM_PACK8;
        case color_format::RGB8_UNORM: return FORMAT_RGB8_UNORM_PACK8;
        case color_format::RGBA8_UNORM: return FORMAT_RGBA8_UNORM_PACK8;
        case color_format::D16_UNORM: return FORMAT_D16_UNORM_PACK16;
        case color_format::X8_D24_UNORM_PACK32: return FORMAT_D24_UNORM_PACK32;
        case color_format::D32_SFLOAT: return FORMAT_D32_SFLOAT_PACK32;
        case color_format::S8_UINT: return FORMAT_R8_SNORM_PACK8;
        case color_format::D16_UNORM_S8_UINT: return FORMAT_D16_UNORM_S8_UINT_PACK32;
        case color_format::D24_UNORM_S8_UINT: return FORMAT_D24_UNORM_S8_UINT_PACK32;
        case color_format::D32_UNORM_S8_UINT: return FORMAT_D32_SFLOAT_S8_UINT_PACK64;
        default: return FORMAT_UNDEFINED;
        }
    }

    static const color_format ToLtseType( const format &aFormat )
    {
        switch( aFormat )
        {
        case FORMAT_R32_SFLOAT_PACK32: return color_format::R32_FLOAT;
        case FORMAT_RG32_SFLOAT_PACK32: return color_format::RG32_FLOAT;
        case FORMAT_RGB32_SFLOAT_PACK32: return color_format::RGB32_FLOAT;
        case FORMAT_RGBA32_SFLOAT_PACK32: return color_format::RGBA32_FLOAT;
        case FORMAT_R16_SFLOAT_PACK16: return color_format::R16_FLOAT;
        case FORMAT_RG16_SFLOAT_PACK16: return color_format::RG16_FLOAT;
        case FORMAT_RGB16_SFLOAT_PACK16: return color_format::RGB16_FLOAT;
        case FORMAT_RGBA16_SFLOAT_PACK16: return color_format::RGBA16_FLOAT;
        case FORMAT_R8_UNORM_PACK8: return color_format::R8_UNORM;
        case FORMAT_RG8_UNORM_PACK8: return color_format::RG8_UNORM;
        case FORMAT_RGB8_UNORM_PACK8: return color_format::RGB8_UNORM;
        case FORMAT_RGBA8_UNORM_PACK8: return color_format::RGBA8_UNORM;
        case FORMAT_D16_UNORM_PACK16: return color_format::D16_UNORM;
        case FORMAT_D24_UNORM_PACK32: return color_format::X8_D24_UNORM_PACK32;
        case FORMAT_D32_SFLOAT_PACK32: return color_format::D32_SFLOAT;
        case FORMAT_R8_SNORM_PACK8: return color_format::S8_UINT;
        case FORMAT_D16_UNORM_S8_UINT_PACK32: return color_format::D16_UNORM_S8_UINT;
        case FORMAT_D24_UNORM_S8_UINT_PACK32: return color_format::D24_UNORM_S8_UINT;
        case FORMAT_D32_SFLOAT_S8_UINT_PACK64: return color_format::D32_UNORM_S8_UINT;
        default: return color_format::UNDEFINED;
        }
    }

    static const target ToGliType( const texture_type &aTextureType )
    {
        switch( aTextureType )
        {
        case texture_type::TEXTURE_3D: return TARGET_3D;
        case texture_type::TEXTURE_2D:
        default: return TARGET_2D;
        }
    }

    static const texture_type ToLtseType( const target &aTextureType )
    {
        switch( aTextureType )
        {
        case TARGET_3D: return texture_type::TEXTURE_3D;
        case TARGET_2D:
        default: return texture_type::TEXTURE_2D;
        }
    }

    static const filter ToGliType( const sampler_filter &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_filter::NEAREST: return FILTER_NEAREST;
        case sampler_filter::LINEAR:
        default: return FILTER_LINEAR;
        }
    }

    static const filter ToGliType( const sampler_mipmap &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_mipmap::NEAREST: return FILTER_NEAREST;
        case sampler_mipmap::LINEAR:
        default: return FILTER_LINEAR;
        }
    }

    static const wrap ToGliType( const sampler_wrapping &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_wrapping::REPEAT: return WRAP_REPEAT;
        case sampler_wrapping::MIRRORED_REPEAT: return WRAP_MIRROR_REPEAT;
        case sampler_wrapping::CLAMP_TO_EDGE: return WRAP_CLAMP_TO_EDGE;
        case sampler_wrapping::MIRROR_CLAMP_TO_BORDER: return WRAP_MIRROR_CLAMP_TO_BORDER;
        case sampler_wrapping::CLAMP_TO_BORDER:
        default: return WRAP_CLAMP_TO_BORDER;
        }
    }

    TextureData::TextureData( texture_create_info_t const &aTextureCreateInfo )
        : mSpec{ aTextureCreateInfo }
    {
        Initialize();
    }

    TextureData::TextureData( texture_create_info_t const &aTextureCreateInfo, image_data_t const &aImageData )
        : mSpec{ aTextureCreateInfo }
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;
        mSpec.mDepth  = 1;

        Initialize();
        std::memcpy( mInternalTexture.data(), aImageData.mPixelData.data(), aImageData.mByteSize );
    }

    TextureData::TextureData( texture_create_info_t const &aTextureCreateInfo, fs::path const &aImagePath )
        : mSpec{ aTextureCreateInfo }
    {
        string_t           lExtension     = aImagePath.extension().string();
        std::set<string_t> lGliExtensions = { ".dds", ".kmg", ".ktx" };

        if( lGliExtensions.find( lExtension ) != lGliExtensions.end() )
        {
            mInternalTexture = load( aImagePath.string() );

            mSpec.mType      = ToLtseType( mInternalTexture.target() );
            mSpec.mFormat    = ToLtseType( mInternalTexture.format() );
            mSpec.mWidth     = mInternalTexture.extent().x;
            mSpec.mHeight    = mInternalTexture.extent().y;
            mSpec.mDepth     = mInternalTexture.extent().z;
            mSpec.mMipLevels = mInternalTexture.levels();
        }
        else
        {
            image_data_t lImageData = LoadImageData( aImagePath );

            mSpec.mType      = texture_type::TEXTURE_2D;
            mSpec.mFormat    = lImageData.mFormat;
            mSpec.mWidth     = lImageData.mWidth;
            mSpec.mHeight    = lImageData.mHeight;
            mSpec.mDepth     = 1;
            mSpec.mMipLevels = 1;

            Initialize();

            std::memcpy( mInternalTexture.data(), lImageData.mPixelData.data(), lImageData.mByteSize );
        }
    }

    TextureData::TextureData( char const *aKTXData, uint32_t aSize )
    {
        mInternalTexture = load( aKTXData, aSize );

        mSpec.mType      = ToLtseType( mInternalTexture.target() );
        mSpec.mFormat    = ToLtseType( mInternalTexture.format() );
        mSpec.mWidth     = mInternalTexture.extent().x;
        mSpec.mHeight    = mInternalTexture.extent().y;
        mSpec.mDepth     = mInternalTexture.extent().z;
        mSpec.mMipLevels = mInternalTexture.levels();
        // mSpec.mSwizzles  = ToLtseType( mInternalTexture.swizzles() );
    }

    void TextureData::SaveTo( fs::path const &aImagePath )
    {
        string_t lExtension = aImagePath.extension().string();

        if( lExtension == ".dds" )
        {
            save_dds( mInternalTexture, aImagePath.string() );
        }
        else if( lExtension == ".kmg" )
        {
            save_kmg( mInternalTexture, aImagePath.string() );
        }
        else if( lExtension == ".ktx" )
        {
            save_ktx( mInternalTexture, aImagePath.string() );
        }
        else
        {
            throw std::runtime_error( "Invalid save extension." );
        }
    }

    vector_t<char> TextureData::Serialize() const
    {
        vector_t<char> lData;
        save_ktx( mInternalTexture, lData );

        return lData;
    }

    void TextureData::Initialize()
    {
        mInternalTexture = texture( ToGliType( mSpec.mType ), ToGliType( mSpec.mFormat ),
                                         extent3d{ mSpec.mWidth, mSpec.mHeight, mSpec.mDepth }, 1, 1, mSpec.mMipLevels );
    }

    TextureData2D::TextureData2D( texture_create_info_t const &aCreateInfo )
        : TextureData( aCreateInfo )
    {
        mInternalTexture2d = texture2d( mInternalTexture );
    }

    TextureData2D::TextureData2D( texture_create_info_t const &aCreateInfo, image_data_t const &aImageData )
        : TextureData( aCreateInfo, aImageData )
    {
        mInternalTexture2d = texture2d( mInternalTexture );

        if( mSpec.mMipLevels > 1 )
        {
            mInternalTexture2d = generate_mipmaps( mInternalTexture2d, FILTER_LINEAR );
        }
    }

    TextureData2D::TextureData2D( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath )
        : TextureData( aCreateInfo, aImagePath )
    {
        mInternalTexture2d = texture2d( mInternalTexture );
    }

    TextureData2D::TextureData2D( char const *aKTXData, uint32_t aSize )
        : TextureData( aKTXData, aSize )
    {
        mInternalTexture2d = texture2d( mInternalTexture );
    }

    image_data_t TextureData2D::GetImageData()
    {
        vector_t<uint8_t> lImageData( (uint8_t*)mInternalTexture2d.data(), ((uint8_t*)mInternalTexture2d.data()) + mInternalTexture2d.size() );
        return { mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ), mInternalTexture2d.size(),
                 std::move( lImageData ) };
    }

    TextureSampler2D::TextureSampler2D( TextureData2D const &aTexture, texture_sampling_info_t const &aSamplingInfo )
        : sampler2d<float>( aTexture.mInternalTexture2d, ToGliType( aSamplingInfo.mWrapping ),
                                 ToGliType( aSamplingInfo.mMipFilter ), ToGliType( aSamplingInfo.mFilter ),
                                 vec4{ aSamplingInfo.mBorderColor[0], aSamplingInfo.mBorderColor[1],
                                            aSamplingInfo.mBorderColor[2], aSamplingInfo.mBorderColor[3] } )
        , mSamplingSpec{ aSamplingInfo }
    {
    }

    TextureDataCubeMap::TextureDataCubeMap( texture_create_info_t const &aCreateInfo )
        : TextureData( aCreateInfo )
    {
        mInternalTextureCubeMap = texture_cube( mInternalTexture );
    }

    TextureDataCubeMap::TextureDataCubeMap( texture_create_info_t const &aCreateInfo, sCubeMapImageData const &aImageData )
        : TextureData( aCreateInfo )
    {
        mInternalTextureCubeMap = texture_cube( mInternalTexture );
    }

    TextureDataCubeMap::TextureDataCubeMap( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath )
        : TextureData( aCreateInfo, aImagePath )
    {
        mInternalTextureCubeMap = texture_cube( mInternalTexture );
    }

    TextureDataCubeMap::TextureDataCubeMap( texture_create_info_t const &aCreateInfo, sCubeMapImagePathData const &aImagePath )
        : TextureData( aCreateInfo )
    {
        mInternalTextureCubeMap = texture_cube( mInternalTexture );
    }

    TextureDataCubeMap::TextureDataCubeMap( vector_t<uint8_t> aKTXData, uint32_t aSize )
        : TextureData( (const char*)aKTXData.data(), aSize )
    {
        mInternalTextureCubeMap = texture_cube( mInternalTexture );
    }

    sCubeMapImageData TextureDataCubeMap::GetImageData()
    {
        sCubeMapImageData lImageData{};

        return std::move(lImageData);
    }



} // namespace SE::Core
