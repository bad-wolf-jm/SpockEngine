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

class TestTextureData : public TextureData
{
  public:
    TestTextureData( sTextureCreateInfo &a_CreateInfo )
        : TextureData( a_CreateInfo ){};
    TestTextureData( sTextureCreateInfo &a_CreateInfo, sImageData &a_ImageData )
        : TextureData( a_CreateInfo, a_ImageData ){};
    TestTextureData( sTextureCreateInfo &a_CreateInfo, fs::path &a_ImagePath )
        : TextureData( a_CreateInfo, a_ImagePath ){};

    gli::texture &GetTexture() { return mInternalTexture; }

    math::ivec2 GetTextureExtent2() { return math::ivec2{ mInternalTexture.extent().x, mInternalTexture.extent().y }; }
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
        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat    = eColorFormat::RGB8_UNORM;
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
            sImageData lImageData = LoadImageData( lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lImageData.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lImageData.mWidth == 256 );
            REQUIRE( lImageData.mHeight == 256 );
            REQUIRE( lImageData.mPixelData.size() != 0 );
        }

        {
            sImageData lImageData = LoadImageData( lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lImageData.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lImageData.mWidth == 256 );
            REQUIRE( lImageData.mHeight == 256 );
            REQUIRE( lImageData.mPixelData.size() != 0 );
        }
    }

    SECTION( "Load abstract textures from file" )
    {
        {
            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData    lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA16_FLOAT );
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
    TestTextureData2D( sTextureCreateInfo &a_CreateInfo )
        : TextureData2D( a_CreateInfo ){};
    TestTextureData2D( sTextureCreateInfo &a_CreateInfo, sImageData &a_ImageData )
        : TextureData2D( a_CreateInfo, a_ImageData ){};
    TestTextureData2D( sTextureCreateInfo &a_CreateInfo, fs::path &a_ImagePath )
        : TextureData2D( a_CreateInfo, a_ImagePath ){};

    gli::texture2d &GetTexture() { return mInternalTexture2d; }

    math::ivec2 GetTextureExtent2() { return math::ivec2{ mInternalTexture.extent().x, mInternalTexture.extent().y }; }
    math::ivec3 GetTextureExtent3()
    {
        return math::ivec3{ mInternalTexture.extent().x, mInternalTexture.extent().y, mInternalTexture.extent().z };
    }
};

TEST_CASE( "Loading 2D textures", "[CORE_CPU_TEXTURES]" )
{
    fs::path lTestDataRoot( "C:\\GitLab\\LTSimulationEngine\\Tests\\Data" );

    SECTION( "Texture creation" )
    {
        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat    = eColorFormat::RGB8_UNORM;
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
            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgb8.jpg" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_srgb8.png" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_snorm.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::UNDEFINED );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba8_unorm.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.dds" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA16_FLOAT );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 256, 256, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            sTextureCreateInfo lTextureCreateInfo{};
            TestTextureData2D  lTexture( lTextureCreateInfo, lTestDataRoot / "kueken7_rgba16_sfloat.ktx" );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA16_FLOAT );
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

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
            REQUIRE( lTexture.GetTexture().size() != 0 );
            REQUIRE( lTexture.GetTextureExtent3() == math::ivec3{ 4, 4, 1 } );
            REQUIRE( lTexture.GetTextureExtent3() ==
                     math::ivec3{ lTexture.mSpec.mWidth, lTexture.mSpec.mHeight, lTexture.mSpec.mDepth } );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::R32_FLOAT );
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

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            sImageData        lRetrievedImageData = lTexture.GetImageData();

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

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TestTextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            sImageData        lRetrievedImageData = lTexture.GetImageData();

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