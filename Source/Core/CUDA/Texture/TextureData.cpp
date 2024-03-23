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

#include "Core/Definitions.h"

#include "TextureData.h"

#include "Core/Logging.h"

#include <filesystem>
#include <fstream>
#include <gli/generate_mipmaps.hpp>

namespace SE::Core
{

    image_data_t LoadImageData( fs::path const &aPath )
    {
        constexpr size_t lComponentCount = 4;

        image_data_t lImageData{};
        int32_t      lActualComponentCount = 0;
        int32_t      lWidth                = 0;
        int32_t      lHeight               = 0;
        size_t       lChannelSize          = 0;

        if( stbi_is_hdr( aPath.string().c_str() ) )
        {
            lChannelSize = 4;

            float *lData = stbi_loadf( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData )
                return image_data_t{};

            lImageData.mFormat = color_format_t::RGBA32_FLOAT;
            lImageData.mPixelData =
                vector_t<uint8_t>( (uint8_t *)lData, ( (uint8_t *)lData ) + ( lWidth * lHeight * sizeof( float ) ) );
        }
        else
        {
            lChannelSize = 1;

            stbi_uc *lData = stbi_load( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData )
                return image_data_t{};

            lImageData.mFormat    = color_format_t::RGBA8_UNORM;
            lImageData.mPixelData = vector_t<uint8_t>( lData, lData + ( lWidth * lHeight * sizeof( float ) ) );
        }

        lImageData.mWidth    = static_cast<size_t>( lWidth );
        lImageData.mHeight   = static_cast<size_t>( lHeight );
        lImageData.mByteSize = lImageData.mWidth * lImageData.mHeight * lComponentCount * lChannelSize;
        return lImageData;
    }

#if 0
    static const gli::format ToGliType( const color_format_t &aFormat )
    {
        switch( aFormat )
        {
        case color_format_t::R32_FLOAT:
            return gli::FORMAT_R32_SFLOAT_PACK32;
        case color_format_t::RG32_FLOAT:
            return gli::FORMAT_RG32_SFLOAT_PACK32;
        case color_format_t::RGB32_FLOAT:
            return gli::FORMAT_RGB32_SFLOAT_PACK32;
        case color_format_t::RGBA32_FLOAT:
            return gli::FORMAT_RGBA32_SFLOAT_PACK32;
        case color_format_t::R16_FLOAT:
            return gli::FORMAT_R16_SFLOAT_PACK16;
        case color_format_t::RG16_FLOAT:
            return gli::FORMAT_RG16_SFLOAT_PACK16;
        case color_format_t::RGB16_FLOAT:
            return gli::FORMAT_RGB16_SFLOAT_PACK16;
        case color_format_t::RGBA16_FLOAT:
            return gli::FORMAT_RGBA16_SFLOAT_PACK16;
        case color_format_t::R8_UNORM:
            return gli::FORMAT_R8_UNORM_PACK8;
        case color_format_t::RG8_UNORM:
            return gli::FORMAT_RG8_UNORM_PACK8;
        case color_format_t::RGB8_UNORM:
            return gli::FORMAT_RGB8_UNORM_PACK8;
        case color_format_t::RGBA8_UNORM:
            return gli::FORMAT_RGBA8_UNORM_PACK8;
        case color_format_t::D16_UNORM:
            return gli::FORMAT_D16_UNORM_PACK16;
        case color_format_t::X8_D24_UNORM_PACK32:
            return gli::FORMAT_D24_UNORM_PACK32;
        case color_format_t::D32_SFLOAT:
            return gli::FORMAT_D32_SFLOAT_PACK32;
        case color_format_t::S8_UINT:
            return gli::FORMAT_R8_SNORM_PACK8;
        case color_format_t::D16_UNORM_S8_UINT:
            return gli::FORMAT_D16_UNORM_S8_UINT_PACK32;
        case color_format_t::D24_UNORM_S8_UINT:
            return gli::FORMAT_D24_UNORM_S8_UINT_PACK32;
        case color_format_t::D32_UNORM_S8_UINT:
            return gli::FORMAT_D32_SFLOAT_S8_UINT_PACK64;
        default:
            return gli::FORMAT_UNDEFINED;
        }
    }

    static const color_format_t ToLtseType( const gli::format &aFormat )
    {
        switch( aFormat )
        {
        case gli::FORMAT_R32_SFLOAT_PACK32:
            return color_format_t::R32_FLOAT;
        case gli::FORMAT_RG32_SFLOAT_PACK32:
            return color_format_t::RG32_FLOAT;
        case gli::FORMAT_RGB32_SFLOAT_PACK32:
            return color_format_t::RGB32_FLOAT;
        case gli::FORMAT_RGBA32_SFLOAT_PACK32:
            return color_format_t::RGBA32_FLOAT;
        case gli::FORMAT_R16_SFLOAT_PACK16:
            return color_format_t::R16_FLOAT;
        case gli::FORMAT_RG16_SFLOAT_PACK16:
            return color_format_t::RG16_FLOAT;
        case gli::FORMAT_RGB16_SFLOAT_PACK16:
            return color_format_t::RGB16_FLOAT;
        case gli::FORMAT_RGBA16_SFLOAT_PACK16:
            return color_format_t::RGBA16_FLOAT;
        case gli::FORMAT_R8_UNORM_PACK8:
            return color_format_t::R8_UNORM;
        case gli::FORMAT_RG8_UNORM_PACK8:
            return color_format_t::RG8_UNORM;
        case gli::FORMAT_RGB8_UNORM_PACK8:
            return color_format_t::RGB8_UNORM;
        case gli::FORMAT_RGBA8_UNORM_PACK8:
            return color_format_t::RGBA8_UNORM;
        case gli::FORMAT_D16_UNORM_PACK16:
            return color_format_t::D16_UNORM;
        case gli::FORMAT_D24_UNORM_PACK32:
            return color_format_t::X8_D24_UNORM_PACK32;
        case gli::FORMAT_D32_SFLOAT_PACK32:
            return color_format_t::D32_SFLOAT;
        case gli::FORMAT_R8_SNORM_PACK8:
            return color_format_t::S8_UINT;
        case gli::FORMAT_D16_UNORM_S8_UINT_PACK32:
            return color_format_t::D16_UNORM_S8_UINT;
        case gli::FORMAT_D24_UNORM_S8_UINT_PACK32:
            return color_format_t::D24_UNORM_S8_UINT;
        case gli::FORMAT_D32_SFLOAT_S8_UINT_PACK64:
            return color_format_t::D32_UNORM_S8_UINT;
        default:
            return color_format_t::UNDEFINED;
        }
    }

    static const gli::target ToGliType( const texture_type_t &aTextureType )
    {
        switch( aTextureType )
        {
        case texture_type_t::TEXTURE_3D:
            return gli::TARGET_3D;
        case texture_type_t::TEXTURE_2D:
        default:
            return gli::TARGET_2D;
        }
    }

    static const texture_type_t ToLtseType( const gli::target &aTextureType )
    {
        switch( aTextureType )
        {
        case gli::TARGET_3D:
            return texture_type_t::TEXTURE_3D;
        case gli::TARGET_2D:
        default:
            return texture_type_t::TEXTURE_2D;
        }
    }

    static const gli::filter ToGliType( const sampler_filter_t &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_filter_t::NEAREST:
            return gli::FILTER_NEAREST;
        case sampler_filter_t::LINEAR:
        default:
            return gli::FILTER_LINEAR;
        }
    }

    static const gli::filter ToGliType( const sampler_mipmap_t &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_mipmap_t::NEAREST:
            return gli::FILTER_NEAREST;
        case sampler_mipmap_t::LINEAR:
        default:
            return gli::FILTER_LINEAR;
        }
    }

    static const gli::wrap ToGliType( const sampler_wrapping_t &aTextureType )
    {
        switch( aTextureType )
        {
        case sampler_wrapping_t::REPEAT:
            return gli::WRAP_REPEAT;
        case sampler_wrapping_t::MIRRORED_REPEAT:
            return gli::WRAP_MIRROR_REPEAT;
        case sampler_wrapping_t::CLAMP_TO_EDGE:
            return gli::WRAP_CLAMP_TO_EDGE;
        case sampler_wrapping_t::MIRROR_CLAMP_TO_BORDER:
            return gli::WRAP_MIRROR_CLAMP_TO_BORDER;
        case sampler_wrapping_t::CLAMP_TO_BORDER:
        default:
            return gli::WRAP_CLAMP_TO_BORDER;
        }
    }

    texture_data_t::texture_data_t( texture_create_info_t const &aTextureCreateInfo )
        : mSpec{ aTextureCreateInfo }
    {
        Initialize();
    }

    texture_data_t::texture_data_t( texture_create_info_t const &aTextureCreateInfo, image_data_t const &aImageData )
        : mSpec{ aTextureCreateInfo }
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;
        mSpec.mDepth  = 1;

        Initialize();
        std::memcpy( mInternalTexture.data(), aImageData.mPixelData.data(), aImageData.mByteSize );
    }

    texture_data_t::texture_data_t( texture_create_info_t const &aTextureCreateInfo, fs::path const &aImagePath )
        : mSpec{ aTextureCreateInfo }
    {
        string_t           lExtension     = aImagePath.extension().string();
        std::set<string_t> lGliExtensions = { ".dds", ".kmg", ".ktx" };

        if( lGliExtensions.find( lExtension ) != lGliExtensions.end() )
        {
            mInternalTexture = gli::load( aImagePath.string() );

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

            mSpec.mType      = texture_type_t::TEXTURE_2D;
            mSpec.mFormat    = lImageData.mFormat;
            mSpec.mWidth     = lImageData.mWidth;
            mSpec.mHeight    = lImageData.mHeight;
            mSpec.mDepth     = 1;
            mSpec.mMipLevels = 1;

            Initialize();

            std::memcpy( mInternalTexture.data(), lImageData.mPixelData.data(), lImageData.mByteSize );
        }
    }

    texture_data_t::texture_data_t( char const *aKTXData, uint32_t aSize )
    {
        mInternalTexture = gli::load( aKTXData, aSize );

        mSpec.mType      = ToLtseType( mInternalTexture.target() );
        mSpec.mFormat    = ToLtseType( mInternalTexture.format() );
        mSpec.mWidth     = mInternalTexture.extent().x;
        mSpec.mHeight    = mInternalTexture.extent().y;
        mSpec.mDepth     = mInternalTexture.extent().z;
        mSpec.mMipLevels = mInternalTexture.levels();
        // mSpec.mSwizzles  = ToLtseType( mInternalTexture.swizzles() );
    }

    void texture_data_t::SaveTo( fs::path const &aImagePath )
    {
        string_t lExtension = aImagePath.extension().string();

        if( lExtension == ".dds" )
        {
            gli::save_dds( mInternalTexture, aImagePath.string() );
        }
        else if( lExtension == ".kmg" )
        {
            gli::save_kmg( mInternalTexture, aImagePath.string() );
        }
        else if( lExtension == ".ktx" )
        {
            gli::save_ktx( mInternalTexture, aImagePath.string() );
        }
        else
        {
            throw std::runtime_error( "Invalid save extension." );
        }
    }

    vector_t<char> texture_data_t::Serialize() const
    {
        vector_t<char> lData;
        gli::save_ktx( mInternalTexture, lData );

        return lData;
    }

    void texture_data_t::Initialize()
    {
        mInternalTexture = gli::texture( ToGliType( mSpec.mType ), ToGliType( mSpec.mFormat ),
                                         gli::extent3d{ mSpec.mWidth, mSpec.mHeight, mSpec.mDepth }, 1, 1, mSpec.mMipLevels );
    }

    texture_data2d_t::texture_data2d_t( texture_create_info_t const &aCreateInfo )
        : texture_data_t( aCreateInfo )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    texture_data2d_t::texture_data2d_t( texture_create_info_t const &aCreateInfo, image_data_t const &aImageData )
        : texture_data_t( aCreateInfo, aImageData )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );

        if( mSpec.mMipLevels > 1 )
        {
            mInternalTexture2d = gli::generate_mipmaps( mInternalTexture2d, gli::FILTER_LINEAR );
        }
    }

    texture_data2d_t::texture_data2d_t( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath )
        : texture_data_t( aCreateInfo, aImagePath )
    {
        // if( mInternalTexture.target() != -1 )
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    texture_data2d_t::texture_data2d_t( char const *aKTXData, uint32_t aSize )
        : texture_data_t( aKTXData, aSize )
    {
        // if( mInternalTexture.target() != -1 )
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    image_data_t texture_data2d_t::GetImageData()
    {
        if( mInternalTexture2d.empty() )
            return { mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ), 0,
                     vector_t<uint8_t>() };

        vector_t<uint8_t> lImageData( (uint8_t *)mInternalTexture2d.data(),
                                      ( (uint8_t *)mInternalTexture2d.data() ) + mInternalTexture2d.size() );
        return { mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ), mInternalTexture2d.size(),
                 std::move( lImageData ) };
    }

    texture_data_sampler2d_t::texture_data_sampler2d_t( texture_data2d_t const        &aTexture,
                                                        texture_sampling_info_t const &aSamplingInfo )
        : gli::sampler2d<float>( aTexture.mInternalTexture2d, ToGliType( aSamplingInfo.mWrapping ),
                                 ToGliType( aSamplingInfo.mMipFilter ), ToGliType( aSamplingInfo.mFilter ),
                                 gli::vec4{ aSamplingInfo.mBorderColor[0], aSamplingInfo.mBorderColor[1],
                                            aSamplingInfo.mBorderColor[2], aSamplingInfo.mBorderColor[3] } )
        , mSamplingSpec{ aSamplingInfo }
    {
    }

    texture_data_cubemap_t::texture_data_cubemap_t( texture_create_info_t const &aCreateInfo )
        : texture_data_t( aCreateInfo )
    {
        mInternalTextureCubeMap = gli::texture_cube( mInternalTexture );
    }

    texture_data_cubemap_t::texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, cubemap_image_data_t const &aImageData )
        : texture_data_t( aCreateInfo )
    {
        mInternalTextureCubeMap = gli::texture_cube( mInternalTexture );
    }

    texture_data_cubemap_t::texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath )
        : texture_data_t( aCreateInfo, aImagePath )
    {
        mInternalTextureCubeMap = gli::texture_cube( mInternalTexture );
    }

    texture_data_cubemap_t::texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, cubemap_image_path_data_t const &aImagePath )
        : texture_data_t( aCreateInfo )
    {
        mInternalTextureCubeMap = gli::texture_cube( mInternalTexture );
    }

    texture_data_cubemap_t::texture_data_cubemap_t( vector_t<uint8_t> aKTXData, uint32_t aSize )
        : texture_data_t( (const char *)aKTXData.data(), aSize )
    {
        mInternalTextureCubeMap = gli::texture_cube( mInternalTexture );
    }

    cubemap_image_data_t texture_data_cubemap_t::GetImageData()
    {
        cubemap_image_data_t lImageData{};

        return std::move( lImageData );
    }
#endif
} // namespace SE::Core
