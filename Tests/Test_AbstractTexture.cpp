#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"

namespace fs = std::filesystem;

using namespace SE::Core;

class TestTextureData : public texture_data_t
{
  public:
    TestTextureData( texture_create_info_t &a_CreateInfo )
        : texture_data_t( a_CreateInfo ){};
    TestTextureData( texture_create_info_t &a_CreateInfo, image_data_t &a_ImageData )
        : texture_data_t( a_CreateInfo, a_ImageData ){};
    TestTextureData( texture_create_info_t &a_CreateInfo, fs::path &a_ImagePath )
        : texture_data_t( a_CreateInfo, a_ImagePath ){};

    gli::texture &GetTexture()
    {
        return mInternalTexture;
    }

    math::ivec2 GetTextureExtent2()
    {
        return math::ivec2{ mInternalTexture.extent().x, mInternalTexture.extent().y };
    }
    math::ivec3 GetTextureExtent3()
    {
        return math::ivec3{ mInternalTexture.extent().x, mInternalTexture.extent().y, mInternalTexture.extent().z };
    }
};

TEST_CASE( "Loading textures", "[CORE_CPU_TEXTURES]" )
{
    fs::path lTestDataRoot( "C:\\GitLab\\SpockEngine\\Tests\\Data" );

    SECTION( "Texture creation" )
    {
        texture_create_info_t lTextureCreateInfo{};
        lTextureCreateInfo.mFormat    = color_format::RGB8_UNORM;
        lTextureCreateInfo.mWidth     = 32;
        lTextureCreateInfo.mHeight    = 64;
        lTextureCreateInfo.mDepth     = 1;
        lTextureCreateInfo.mMipLevels = 1;

        TestTextureData lTexture( lTextureCreateInfo );
        REQUIRE( lTexture.GetTexture().size() != 0 );
        REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 32, 64, 1 } );
    }

    SECTION( "Load images from file" )
    {
        {
            image_data_t lImageData = LoadImageData( lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lImageData.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lImageData.mWidth == 256 );
            REQUIRE( lImageData.mHeight == 256 );
            REQUIRE( lImageData.mPixelData.size() != 0 );
        }

        {
            image_data_t lImageData = LoadImageData( lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lImageData.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lImageData.mWidth == 256 );
            REQUIRE( lImageData.mHeight == 256 );
            REQUIRE( lImageData.mPixelData.size() != 0 );
        }
    }

    SECTION( "Load abstract textures from file" )
    {
        {
            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }
    }
}

class TestTextureData2D : public texture_data2d_t
{
  public:
    TestTextureData2D( texture_create_info_t &a_CreateInfo )
        : texture_data2d_t( a_CreateInfo ){};
    TestTextureData2D( texture_create_info_t &a_CreateInfo, image_data_t &a_ImageData )
        : texture_data2d_t( a_CreateInfo, a_ImageData ){};
    TestTextureData2D( texture_create_info_t &a_CreateInfo, fs::path &a_ImagePath )
        : texture_data2d_t( a_CreateInfo, a_ImagePath ){};

    gli::texture2d &GetTexture()
    {
        return mInternalTexture2d;
    }

    math::ivec2 GetTextureExtent2()
    {
        return math::ivec2{ mInternalTexture.extent().x, mInternalTexture.extent().y };
    }
    math::ivec3 GetTextureExtent3()
    {
        return math::ivec3{ mInternalTexture.extent().x, mInternalTexture.extent().y, mInternalTexture.extent().z };
    }
};

TEST_CASE( "Loading 2D textures", "[CORE_CPU_TEXTURES]" )
{
    fs::path lTestDataRoot( "C:\\GitLab\\SpockEngine\\Tests\\Data" );

    SECTION( "Texture creation" )
    {
        texture_create_info_t lTextureCreateInfo{};
        lTextureCreateInfo.mFormat    = color_format::RGB8_UNORM;
        lTextureCreateInfo.mWidth     = 32;
        lTextureCreateInfo.mHeight    = 64;
        lTextureCreateInfo.mDepth     = 1;
        lTextureCreateInfo.mMipLevels = 1;

        TestTextureData2D lTexture( lTextureCreateInfo );
        REQUIRE( lTexture.GetTexture().size() != 0 );
        REQUIRE( lTexture.GetTexture().extent().x == 32 );
        REQUIRE( lTexture.GetTexture().extent().y == 64 );
    }

    SECTION( "Load 2D textures from file" )
    {
        {
            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }
    }

    SECTION( "Load 2D textures from data" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                        0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                        0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 4, 4, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            REQUIRE( lTexture.mSpec.mFormat == color_format::R32_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 4, 4, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }
    }

    SECTION( "Retrieve image data" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                        0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                        0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            image_data_t        lRetrievedImageData = lTexture.GetImageData();

            REQUIRE( lImageDataStruct.mFormat == lRetrievedImageData.mFormat );
            REQUIRE( lImageDataStruct.mWidth == lRetrievedImageData.mWidth );
            REQUIRE( lImageDataStruct.mHeight == lRetrievedImageData.mHeight );
            REQUIRE( lImageDataStruct.mByteSize == lRetrievedImageData.mByteSize );
            bool lEqual = true;
            for( uint32_t i = 0; i < lImageDataStruct.mByteSize; i++ )
                lEqual = lEqual && ( lImageDataStruct.mPixelData[i] == lRetrievedImageData.mPixelData[i] );
            REQUIRE( lEqual );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            image_data_t        lRetrievedImageData = lTexture.GetImageData();

            REQUIRE( lImageDataStruct.mFormat == lRetrievedImageData.mFormat );
            REQUIRE( lImageDataStruct.mWidth == lRetrievedImageData.mWidth );
            REQUIRE( lImageDataStruct.mHeight == lRetrievedImageData.mHeight );
            REQUIRE( lImageDataStruct.mByteSize == lRetrievedImageData.mByteSize );
            bool lEqual = true;
            for( uint32_t i = 0; i < lImageDataStruct.mByteSize; i++ )
                lEqual = lEqual && ( lImageDataStruct.mPixelData[i] == lRetrievedImageData.mPixelData[i] );
            REQUIRE( lEqual );
        }
    }
}