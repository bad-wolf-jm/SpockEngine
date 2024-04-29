#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"

namespace fs = std::filesystem;

using namespace numlua::core;

class TestTextureData : public TextureData
{
  public:
    TestTextureData( texture_create_info_t &createInfo )
        : TextureData( createInfo ){};
    TestTextureData( texture_create_info_t &createInfo, image_data_t &a_ImageData )
        : TextureData( createInfo, a_ImageData ){};
    TestTextureData( texture_create_info_t &createInfo, fs::path &a_ImagePath )
        : TextureData( createInfo, a_ImagePath ){};

    texture &GetTexture()
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
    fs::path testDataRoot( "C:\\GitLab\\SpockEngine\\Tests\\Data" );

    SECTION( "Texture creation" )
    {
        texture_create_info_t textureCreateInfo{};
        textureCreateInfo.mFormat    = color_format::RGB8_UNORM;
        textureCreateInfo.mWidth     = 32;
        textureCreateInfo.mHeight    = 64;
        textureCreateInfo.mDepth     = 1;
        textureCreateInfo.mMipLevels = 1;

        TestTextureData lTexture( textureCreateInfo );
        REQUIRE( lTexture.GetTexture().size() != 0 );
        REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 32, 64, 1 } );
    }

    SECTION( "Load images from file" )
    {
        {
            image_data_t imageData = LoadImageData( testDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( imageData.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( imageData.mWidth == 256 );
            REQUIRE( imageData.mHeight == 256 );
            REQUIRE( imageData.mPixelData.size() != 0 );
        }

        {
            image_data_t imageData = LoadImageData( testDataRoot / "kueken7_srgb8.png" );
            REQUIRE( imageData.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( imageData.mWidth == 256 );
            REQUIRE( imageData.mHeight == 256 );
            REQUIRE( imageData.mPixelData.size() != 0 );
        }
    }

    SECTION( "Load abstract textures from file" )
    {
        {
            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( textureCreateInfo, testDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( textureCreateInfo, testDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData       lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData       lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData       lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData       lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData       lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba16_sfloat.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }
    }
}

class TestTextureData2D : public TextureData2D
{
  public:
    TestTextureData2D( texture_create_info_t &createInfo )
        : TextureData2D( createInfo ){};
    TestTextureData2D( texture_create_info_t &createInfo, image_data_t &a_ImageData )
        : TextureData2D( createInfo, a_ImageData ){};
    TestTextureData2D( texture_create_info_t &createInfo, fs::path &a_ImagePath )
        : TextureData2D( createInfo, a_ImagePath ){};

    texture2d &GetTexture()
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
    fs::path testDataRoot( "C:\\GitLab\\SpockEngine\\Tests\\Data" );

    SECTION( "Texture creation" )
    {
        texture_create_info_t textureCreateInfo{};
        textureCreateInfo.mFormat    = color_format::RGB8_UNORM;
        textureCreateInfo.mWidth     = 32;
        textureCreateInfo.mHeight    = 64;
        textureCreateInfo.mDepth     = 1;
        textureCreateInfo.mMipLevels = 1;

        TestTextureData2D lTexture( textureCreateInfo );
        REQUIRE( lTexture.GetTexture().size() != 0 );
        REQUIRE( lTexture.GetTexture().extent().x == 32 );
        REQUIRE( lTexture.GetTexture().extent().y == 64 );
    }

    SECTION( "Load 2D textures from file" )
    {
        {
            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, testDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, testDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData2D     lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData2D     lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData2D     lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData2D     lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            texture_create_info_t textureCreateInfo{};
            TestTextureData2D     lTexture( textureCreateInfo, testDataRoot / "kueken7_rgba16_sfloat.ktx" );
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
            uint32_t imageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                       0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            image_data_t imageDataStruct{};
            imageDataStruct.mFormat    = color_format::RGBA8_UNORM;
            imageDataStruct.mWidth     = 4;
            imageDataStruct.mHeight    = 4;
            imageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            imageDataStruct.mPixelData = std::vector<uint8_t>( reinterpret_cast<uint8_t *>( imageData ),
                                                               reinterpret_cast<uint8_t *>( imageData ) + imageDataStruct.mByteSize );

            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, imageDataStruct );
            REQUIRE( lTexture.mSpec.mFormat == color_format::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 4, 4, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            float imageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            image_data_t imageDataStruct{};
            imageDataStruct.mFormat    = color_format::R32_FLOAT;
            imageDataStruct.mWidth     = 4;
            imageDataStruct.mHeight    = 4;
            imageDataStruct.mByteSize  = 16 * sizeof( float );
            imageDataStruct.mPixelData = std::vector<uint8_t>( reinterpret_cast<uint8_t *>( imageData ),
                                                               reinterpret_cast<uint8_t *>( imageData ) + imageDataStruct.mByteSize );

            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, imageDataStruct );
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
            uint32_t imageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                       0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            image_data_t imageDataStruct{};
            imageDataStruct.mFormat    = color_format::RGBA8_UNORM;
            imageDataStruct.mWidth     = 4;
            imageDataStruct.mHeight    = 4;
            imageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            imageDataStruct.mPixelData = std::vector<uint8_t>( reinterpret_cast<uint8_t *>( imageData ),
                                                               reinterpret_cast<uint8_t *>( imageData ) + imageDataStruct.mByteSize );

            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, imageDataStruct );
            image_data_t      lRetrievedImageData = lTexture.GetImageData();

            REQUIRE( imageDataStruct.mFormat == lRetrievedImageData.mFormat );
            REQUIRE( imageDataStruct.mWidth == lRetrievedImageData.mWidth );
            REQUIRE( imageDataStruct.mHeight == lRetrievedImageData.mHeight );
            REQUIRE( imageDataStruct.mByteSize == lRetrievedImageData.mByteSize );
            bool lEqual = true;
            for( uint32_t i = 0; i < imageDataStruct.mByteSize; i++ )
                lEqual = lEqual && ( imageDataStruct.mPixelData[i] == lRetrievedImageData.mPixelData[i] );
            REQUIRE( lEqual );
        }

        {
            float imageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            image_data_t imageDataStruct{};
            imageDataStruct.mFormat    = color_format::R32_FLOAT;
            imageDataStruct.mWidth     = 4;
            imageDataStruct.mHeight    = 4;
            imageDataStruct.mByteSize  = 16 * sizeof( float );
            imageDataStruct.mPixelData = std::vector<uint8_t>( reinterpret_cast<uint8_t *>( imageData ),
                                                               reinterpret_cast<uint8_t *>( imageData ) + imageDataStruct.mByteSize );

            texture_create_info_t textureCreateInfo{};
            textureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( textureCreateInfo, imageDataStruct );
            image_data_t      lRetrievedImageData = lTexture.GetImageData();

            REQUIRE( imageDataStruct.mFormat == lRetrievedImageData.mFormat );
            REQUIRE( imageDataStruct.mWidth == lRetrievedImageData.mWidth );
            REQUIRE( imageDataStruct.mHeight == lRetrievedImageData.mHeight );
            REQUIRE( imageDataStruct.mByteSize == lRetrievedImageData.mByteSize );
            bool lEqual = true;
            for( uint32_t i = 0; i < imageDataStruct.mByteSize; i++ )
                lEqual = lEqual && ( imageDataStruct.mPixelData[i] == lRetrievedImageData.mPixelData[i] );
            REQUIRE( lEqual );
        }
    }
}