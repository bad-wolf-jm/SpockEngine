/// @file   TextureData.cpp
///
/// @brief  Implementation file for texture data
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#include <set>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

#include "Core/Core.h"
#include "Core/Memory.h"

#include "TextureData.h"

#include "Core/Logging.h"

#include <filesystem>
#include <fstream>
#include <gli/generate_mipmaps.hpp>

namespace SE::Core
{

    sImageData LoadImageData( fs::path const &aPath )
    {
        constexpr size_t lComponentCount = 4;

        sImageData lImageData{};
        int32_t    lActualComponentCount = 0;
        int32_t    lWidth                = 0;
        int32_t    lHeight               = 0;
        size_t     lChannelSize          = 0;

        if( aPath.extension().string() == ".bin" )
        {
            std::ifstream l_InputFile;
            l_InputFile.open( aPath, std::ios::in | std::ios::binary );
            uint32_t l_Height, l_Width;
            l_InputFile.seekg( 0, std::ios::beg );
            l_InputFile.read( (char *)&l_Height, sizeof( uint32_t ) );
            l_InputFile.read( (char *)&l_Width, sizeof( uint32_t ) );
            l_InputFile.seekg( 2 * sizeof( uint32_t ), std::ios::beg );

            float *l_Value = (float *)malloc( l_Height * l_Width * sizeof( float ) );
            l_InputFile.read( (char *)l_Value, l_Height * l_Width * sizeof( float ) );

            lImageData.mWidth     = static_cast<size_t>( l_Width );
            lImageData.mHeight    = static_cast<size_t>( l_Height );
            lImageData.mFormat    = eColorFormat::R32_FLOAT;
            lImageData.mByteSize  = l_Height * l_Width * sizeof( float );
            lImageData.mPixelData = (uint8_t *)l_Value;
            return lImageData;
        }

        if( stbi_is_hdr( aPath.string().c_str() ) )
        {
            lChannelSize = 4;

            float *lData = stbi_loadf( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData ) return sImageData{};

            lImageData.mFormat    = eColorFormat::RGBA32_FLOAT;
            lImageData.mPixelData = reinterpret_cast<uint8_t *>( lData );
        }
        else
        {
            lChannelSize = 1;

            stbi_uc *lData = stbi_load( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData ) return sImageData{};

            lImageData.mFormat    = eColorFormat::RGBA8_UNORM;
            lImageData.mPixelData = reinterpret_cast<uint8_t *>( lData );
        }

        lImageData.mWidth    = static_cast<size_t>( lWidth );
        lImageData.mHeight   = static_cast<size_t>( lHeight );
        lImageData.mByteSize = lImageData.mWidth * lImageData.mHeight * lComponentCount * lChannelSize;
        return lImageData;
    }

    static const gli::format ToGliType( const eColorFormat &aFormat )
    {
        switch( aFormat )
        {
        case eColorFormat::R32_FLOAT:
            return gli::FORMAT_R32_SFLOAT_PACK32;
        case eColorFormat::RG32_FLOAT:
            return gli::FORMAT_RG32_SFLOAT_PACK32;
        case eColorFormat::RGB32_FLOAT:
            return gli::FORMAT_RGB32_SFLOAT_PACK32;
        case eColorFormat::RGBA32_FLOAT:
            return gli::FORMAT_RGBA32_SFLOAT_PACK32;
        case eColorFormat::R16_FLOAT:
            return gli::FORMAT_R16_SFLOAT_PACK16;
        case eColorFormat::RG16_FLOAT:
            return gli::FORMAT_RG16_SFLOAT_PACK16;
        case eColorFormat::RGB16_FLOAT:
            return gli::FORMAT_RGB16_SFLOAT_PACK16;
        case eColorFormat::RGBA16_FLOAT:
            return gli::FORMAT_RGBA16_SFLOAT_PACK16;
        case eColorFormat::R8_UNORM:
            return gli::FORMAT_R8_UNORM_PACK8;
        case eColorFormat::RG8_UNORM:
            return gli::FORMAT_RG8_UNORM_PACK8;
        case eColorFormat::RGB8_UNORM:
            return gli::FORMAT_RGB8_UNORM_PACK8;
        case eColorFormat::RGBA8_UNORM:
            return gli::FORMAT_RGBA8_UNORM_PACK8;
        case eColorFormat::D16_UNORM:
            return gli::FORMAT_D16_UNORM_PACK16;
        case eColorFormat::X8_D24_UNORM_PACK32:
            return gli::FORMAT_D24_UNORM_PACK32;
        case eColorFormat::D32_SFLOAT:
            return gli::FORMAT_D32_SFLOAT_PACK32;
        case eColorFormat::S8_UINT:
            return gli::FORMAT_R8_SNORM_PACK8;
        case eColorFormat::D16_UNORM_S8_UINT:
            return gli::FORMAT_D16_UNORM_S8_UINT_PACK32;
        case eColorFormat::D24_UNORM_S8_UINT:
            return gli::FORMAT_D24_UNORM_S8_UINT_PACK32;
        case eColorFormat::D32_UNORM_S8_UINT:
            return gli::FORMAT_D32_SFLOAT_S8_UINT_PACK64;
        default:
            return gli::FORMAT_UNDEFINED;
        }
    }

    static const eColorFormat ToLtseType( const gli::format &aFormat )
    {
        switch( aFormat )
        {
        case gli::FORMAT_R32_SFLOAT_PACK32:
            return eColorFormat::R32_FLOAT;
        case gli::FORMAT_RG32_SFLOAT_PACK32:
            return eColorFormat::RG32_FLOAT;
        case gli::FORMAT_RGB32_SFLOAT_PACK32:
            return eColorFormat::RGB32_FLOAT;
        case gli::FORMAT_RGBA32_SFLOAT_PACK32:
            return eColorFormat::RGBA32_FLOAT;
        case gli::FORMAT_R16_SFLOAT_PACK16:
            return eColorFormat::R16_FLOAT;
        case gli::FORMAT_RG16_SFLOAT_PACK16:
            return eColorFormat::RG16_FLOAT;
        case gli::FORMAT_RGB16_SFLOAT_PACK16:
            return eColorFormat::RGB16_FLOAT;
        case gli::FORMAT_RGBA16_SFLOAT_PACK16:
            return eColorFormat::RGBA16_FLOAT;
        case gli::FORMAT_R8_UNORM_PACK8:
            return eColorFormat::R8_UNORM;
        case gli::FORMAT_RG8_UNORM_PACK8:
            return eColorFormat::RG8_UNORM;
        case gli::FORMAT_RGB8_UNORM_PACK8:
            return eColorFormat::RGB8_UNORM;
        case gli::FORMAT_RGBA8_UNORM_PACK8:
            return eColorFormat::RGBA8_UNORM;
        case gli::FORMAT_D16_UNORM_PACK16:
            return eColorFormat::D16_UNORM;
        case gli::FORMAT_D24_UNORM_PACK32:
            return eColorFormat::X8_D24_UNORM_PACK32;
        case gli::FORMAT_D32_SFLOAT_PACK32:
            return eColorFormat::D32_SFLOAT;
        case gli::FORMAT_R8_SNORM_PACK8:
            return eColorFormat::S8_UINT;
        case gli::FORMAT_D16_UNORM_S8_UINT_PACK32:
            return eColorFormat::D16_UNORM_S8_UINT;
        case gli::FORMAT_D24_UNORM_S8_UINT_PACK32:
            return eColorFormat::D24_UNORM_S8_UINT;
        case gli::FORMAT_D32_SFLOAT_S8_UINT_PACK64:
            return eColorFormat::D32_UNORM_S8_UINT;
        default:
            return eColorFormat::UNDEFINED;
        }
    }

    static const gli::swizzle ToGliType( const eSwizzleComponent &aSwizzleComponent )
    {
        switch( aSwizzleComponent )
        {
        case eSwizzleComponent::ONE:
            return gli::SWIZZLE_ONE;
        case eSwizzleComponent::R:
            return gli::SWIZZLE_RED;
        case eSwizzleComponent::G:
            return gli::SWIZZLE_GREEN;
        case eSwizzleComponent::B:
            return gli::SWIZZLE_BLUE;
        case eSwizzleComponent::A:
            return gli::SWIZZLE_ALPHA;
        case eSwizzleComponent::ZERO:
        default:
            return gli::SWIZZLE_ZERO;
        }
    }

    static const eSwizzleComponent ToLtseType( const gli::swizzle &aSwizzleComponent )
    {
        switch( aSwizzleComponent )
        {
        case gli::SWIZZLE_ONE:
            return eSwizzleComponent::ONE;
        case gli::SWIZZLE_RED:
            return eSwizzleComponent::R;
        case gli::SWIZZLE_GREEN:
            return eSwizzleComponent::G;
        case gli::SWIZZLE_BLUE:
            return eSwizzleComponent::B;
        case gli::SWIZZLE_ALPHA:
            return eSwizzleComponent::A;
        case gli::SWIZZLE_ZERO:
        default:
            return eSwizzleComponent::ZERO;
        }
    }

    static const gli::target ToGliType( const eTextureType &aTextureType )
    {
        switch( aTextureType )
        {
        case eTextureType::TEXTURE_3D:
            return gli::TARGET_3D;
        case eTextureType::TEXTURE_2D:
        default:
            return gli::TARGET_2D;
        }
    }

    static const eTextureType ToLtseType( const gli::target &aTextureType )
    {
        switch( aTextureType )
        {
        case gli::TARGET_3D:
            return eTextureType::TEXTURE_3D;
        case gli::TARGET_2D:
        default:
            return eTextureType::TEXTURE_2D;
        }
    }

    static const gli::swizzles ToGliType( const sSwizzleTransform &aSwizzleTransform )
    {
        return gli::swizzles( ToGliType( aSwizzleTransform.mR ), ToGliType( aSwizzleTransform.mG ), ToGliType( aSwizzleTransform.mB ),
            ToGliType( aSwizzleTransform.mA ) );
    }

    static const sSwizzleTransform ToLtseType( const gli::swizzles &aSwizzleTransform )
    {
        return sSwizzleTransform{ ToLtseType( aSwizzleTransform.r ), ToLtseType( aSwizzleTransform.g ),
            ToLtseType( aSwizzleTransform.b ), ToLtseType( aSwizzleTransform.a ) };
    }

    static const gli::filter ToGliType( const eSamplerFilter &aTextureType )
    {
        switch( aTextureType )
        {
        case eSamplerFilter::NEAREST:
            return gli::FILTER_NEAREST;
        case eSamplerFilter::LINEAR:
        default:
            return gli::FILTER_LINEAR;
        }
    }

    static const gli::filter ToGliType( const eSamplerMipmap &aTextureType )
    {
        switch( aTextureType )
        {
        case eSamplerMipmap::NEAREST:
            return gli::FILTER_NEAREST;
        case eSamplerMipmap::LINEAR:
        default:
            return gli::FILTER_LINEAR;
        }
    }

    static const gli::wrap ToGliType( const eSamplerWrapping &aTextureType )
    {
        switch( aTextureType )
        {
        case eSamplerWrapping::REPEAT:
            return gli::WRAP_REPEAT;
        case eSamplerWrapping::MIRRORED_REPEAT:
            return gli::WRAP_MIRROR_REPEAT;
        case eSamplerWrapping::CLAMP_TO_EDGE:
            return gli::WRAP_CLAMP_TO_EDGE;
        case eSamplerWrapping::MIRROR_CLAMP_TO_BORDER:
            return gli::WRAP_MIRROR_CLAMP_TO_BORDER;
        case eSamplerWrapping::CLAMP_TO_BORDER:
        default:
            return gli::WRAP_CLAMP_TO_BORDER;
        }
    }

    TextureData::TextureData( TextureData::sCreateInfo const &aTextureCreateInfo )
        : mSpec{ aTextureCreateInfo }
    {
        Initialize();
    }

    TextureData::TextureData( TextureData::sCreateInfo const &aTextureCreateInfo, sImageData const &aImageData )
        : mSpec{ aTextureCreateInfo }
    {
        mSpec.mFormat = aImageData.mFormat;
        mSpec.mWidth  = aImageData.mWidth;
        mSpec.mHeight = aImageData.mHeight;
        mSpec.mDepth  = 1;

        Initialize();
        std::memcpy( mInternalTexture.data(), aImageData.mPixelData, aImageData.mByteSize );
    }

    TextureData::TextureData( TextureData::sCreateInfo const &aTextureCreateInfo, fs::path const &aImagePath )
        : mSpec{ aTextureCreateInfo }
    {
        std::string           lExtension     = aImagePath.extension().string();
        std::set<std::string> lGliExtensions = { ".dds", ".kmg", ".ktx" };

        if( lGliExtensions.find( lExtension ) != lGliExtensions.end() )
        {
            mInternalTexture = gli::load( aImagePath.string() );

            mSpec.mType      = ToLtseType( mInternalTexture.target() );
            mSpec.mFormat    = ToLtseType( mInternalTexture.format() );
            mSpec.mWidth     = mInternalTexture.extent().x;
            mSpec.mHeight    = mInternalTexture.extent().y;
            mSpec.mDepth     = mInternalTexture.extent().z;
            mSpec.mMipLevels = mInternalTexture.levels();
            mSpec.mSwizzles  = ToLtseType( mInternalTexture.swizzles() );
        }
        else
        {
            sImageData lImageData = LoadImageData( aImagePath );

            mSpec.mType      = eTextureType::TEXTURE_2D;
            mSpec.mFormat    = lImageData.mFormat;
            mSpec.mWidth     = lImageData.mWidth;
            mSpec.mHeight    = lImageData.mHeight;
            mSpec.mDepth     = 1;
            mSpec.mMipLevels = 1;

            // SE::Logging::Info("File: {} -- Format: {}", aImagePath.string(), (uint32_t)lImageData.mFormat);
            Initialize();

            std::memcpy( mInternalTexture.data(), lImageData.mPixelData, lImageData.mByteSize );
        }
    }

    TextureData::TextureData( char const *aKTXData, uint32_t aSize )
    {
        mInternalTexture = gli::load( aKTXData, aSize );

        mSpec.mType      = ToLtseType( mInternalTexture.target() );
        mSpec.mFormat    = ToLtseType( mInternalTexture.format() );
        mSpec.mWidth     = mInternalTexture.extent().x;
        mSpec.mHeight    = mInternalTexture.extent().y;
        mSpec.mDepth     = mInternalTexture.extent().z;
        mSpec.mMipLevels = mInternalTexture.levels();
        mSpec.mSwizzles  = ToLtseType( mInternalTexture.swizzles() );
    }

    void TextureData::SaveTo( fs::path const &aImagePath )
    {
        std::string lExtension = aImagePath.extension().string();

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

    std::vector<char> TextureData::Serialize() const
    {
        std::vector<char> lData;
        gli::save_ktx( mInternalTexture, lData );

        return lData;
    }

    void TextureData::Initialize()
    {
        mInternalTexture = gli::texture( ToGliType( mSpec.mType ), ToGliType( mSpec.mFormat ),
            gli::extent3d{ mSpec.mWidth, mSpec.mHeight, mSpec.mDepth }, 1, 1, mSpec.mMipLevels, ToGliType( mSpec.mSwizzles ) );
    }

    TextureData2D::TextureData2D( TextureData::sCreateInfo const &aCreateInfo )
        : TextureData( aCreateInfo )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    TextureData2D::TextureData2D( TextureData::sCreateInfo const &aCreateInfo, sImageData const &aImageData )
        : TextureData( aCreateInfo, aImageData )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );

        if( mSpec.mMipLevels > 1 )
        {
            mInternalTexture2d = gli::generate_mipmaps( mInternalTexture2d, gli::FILTER_LINEAR );
        }
    }

    TextureData2D::TextureData2D( TextureData::sCreateInfo const &aCreateInfo, fs::path const &aImagePath )
        : TextureData( aCreateInfo, aImagePath )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    TextureData2D::TextureData2D( char const *aKTXData, uint32_t aSize )
        : TextureData( aKTXData, aSize )
    {
        mInternalTexture2d = gli::texture2d( mInternalTexture );
    }

    sImageData TextureData2D::GetImageData()
    {
        return { mSpec.mFormat, static_cast<size_t>( mSpec.mWidth ), static_cast<size_t>( mSpec.mHeight ), mInternalTexture2d.size(),
            reinterpret_cast<uint8_t *>( mInternalTexture2d.data() ) };
    }

    TextureSampler2D::TextureSampler2D( TextureData2D const &aTexture, sTextureSamplingInfo const &aSamplingInfo )
        : gli::sampler2d<float>( aTexture.mInternalTexture2d, ToGliType( aSamplingInfo.mWrapping ), ToGliType( aSamplingInfo.mMip ),
              ToGliType( aSamplingInfo.mMinification ),
              gli::vec4{ aSamplingInfo.mBorderColor[0], aSamplingInfo.mBorderColor[1], aSamplingInfo.mBorderColor[2],
                  aSamplingInfo.mBorderColor[3] } )
        , mSamplingSpec{ aSamplingInfo }
    {
    }

    std::array<float, 4> TextureSampler2D::Fetch( float x, float y )
    {
        gli::vec4 lColor = texture_lod(
            gli::sampler2d<float>::normalized_type( x / mSamplingSpec.mScaling[0], y / mSamplingSpec.mScaling[1] ), 0.0f );
        return std::array<float, 4>{ lColor.x, lColor.y, lColor.z, lColor.w };
    }

} // namespace SE::Core
