#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "TestUtils.h"

#include "Core/Math/Types.h"
#include "Core/TextureData.h"
#include "Core/TextureTypes.h"

#include "Cuda/Texture2D.h"

namespace fs = std::filesystem;

using namespace LTSE;
using namespace LTSE::Core;
using namespace LTSE::Cuda;

TEST_CASE( "Loading Cuda 2D textures", "[CORE_CUDA_TEXTURES]" )
{
    fs::path lTestDataRoot( "C:\\GitLab\\LTSimulationEngine\\Tests\\Data" );

    SECTION( "Load 2D textures from data" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            std::vector<uint8_t> lPixelDataVector( 16 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ), 16 * sizeof( uint32_t ) );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = eColorFormat::RGBA8_UNORM;
            lTextureCreateInfo.mWidth  = 4;
            lTextureCreateInfo.mHeight = 4;
            Texture2D lTexture( lTextureCreateInfo, lPixelDataVector );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };
            std::vector<uint8_t> lPixelDataVector( 16 * sizeof( float ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ), 16 * sizeof( float ) );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = eColorFormat::R32_FLOAT;
            lTextureCreateInfo.mWidth  = 4;
            lTextureCreateInfo.mHeight = 4;
            Texture2D lTexture( lTextureCreateInfo, lPixelDataVector );
            REQUIRE( lTexture.mSpec.mFormat == eColorFormat::R32_FLOAT );
        }
    }

    SECTION( "Create cuda texture sImageData" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = reinterpret_cast<uint8_t *>( lImageData );

            sTextureCreateInfo lCudaTextureCreateInfo{};
            Texture2D lCudaTexture( lCudaTextureCreateInfo, lImageDataStruct );
            REQUIRE( lCudaTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = reinterpret_cast<uint8_t *>( lImageData );

            sTextureCreateInfo lCudaTextureCreateInfo{};
            Texture2D lCudaTexture( lCudaTextureCreateInfo, lImageDataStruct );
            REQUIRE( lCudaTexture.mSpec.mFormat == eColorFormat::R32_FLOAT );
        }
    }

    SECTION( "Load cuda 2D textures from data" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = reinterpret_cast<uint8_t *>( lImageData );

            TextureData::sCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );

            sTextureCreateInfo lCudaTextureCreateInfo{};
            Texture2D lCudaTexture( lCudaTextureCreateInfo, lTexture.GetImageData() );
            REQUIRE( lCudaTexture.mSpec.mFormat == eColorFormat::RGBA8_UNORM );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            sImageData lImageDataStruct{};
            lImageDataStruct.mFormat    = eColorFormat::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = reinterpret_cast<uint8_t *>( lImageData );

            TextureData::sCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mMipLevels = 1;
            TextureData2D lTexture( lTextureCreateInfo, lImageDataStruct );
            sTextureCreateInfo lCudaTextureCreateInfo{};
            Texture2D lCudaTexture( lCudaTextureCreateInfo, lTexture.GetImageData() );
            REQUIRE( lCudaTexture.mSpec.mFormat == eColorFormat::R32_FLOAT );
        }
    }

    SECTION( "Sampling 2D textures" )
    {
        {
            uint32_t lImageData[9] = { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF };

            std::vector<uint8_t> lPixelDataVector( 9 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ), 9 * sizeof( uint32_t ) );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mFormat           = eColorFormat::RGBA8_UNORM;
            lTextureCreateInfo.mWidth            = 3;
            lTextureCreateInfo.mHeight           = 3;
            lTextureCreateInfo.mNormalizedValues = true;
            Ref<Texture2D> lTexture              = New<Texture2D>( lTextureCreateInfo, lPixelDataVector );

            sTextureSamplingInfo lSamplingInfo{};
            Cuda::TextureSampler2D lTextureSampler = Cuda::TextureSampler2D( lTexture, lSamplingInfo );

            REQUIRE( lTextureSampler.mSamplingSpec.mMinification == eSamplerFilter::LINEAR );
            REQUIRE( lTextureSampler.mSamplingSpec.mMagnification == eSamplerFilter::LINEAR );
            REQUIRE( lTextureSampler.mSamplingSpec.mWrapping == eSamplerWrapping::CLAMP_TO_BORDER );
            REQUIRE( lTextureSampler.mSamplingSpec.mBorderColor[0] == 0.0f );
            REQUIRE( lTextureSampler.mSamplingSpec.mBorderColor[1] == 0.0f );
            REQUIRE( lTextureSampler.mSamplingSpec.mBorderColor[2] == 0.0f );
            REQUIRE( lTextureSampler.mSamplingSpec.mBorderColor[3] == 0.0f );

            REQUIRE( TestUtils::VectorEqual( lTextureSampler.mSamplingSpec.mScaling, std::array<float, 2>{ 1.0f, 1.0f } ) );
        }

        {
            uint32_t lImageData[9] = { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF };

            std::vector<uint8_t> lPixelDataVector( 9 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ), 9 * sizeof( uint32_t ) );

            sTextureCreateInfo lTextureCreateInfo{};
            lTextureCreateInfo.mFormat           = eColorFormat::RGBA8_UNORM;
            lTextureCreateInfo.mWidth            = 3;
            lTextureCreateInfo.mHeight           = 3;
            lTextureCreateInfo.mNormalizedValues = true;
            Ref<Texture2D> lTexture              = New<Texture2D>( lTextureCreateInfo, lPixelDataVector );

            sTextureSamplingInfo lSamplingInfo{};
            lSamplingInfo.mScaling                 = std::array<float, 2>{ 3.0f, 4.0f };
            Cuda::TextureSampler2D lTextureSampler = Cuda::TextureSampler2D( lTexture, lSamplingInfo );

            REQUIRE( TestUtils::VectorEqual( lTextureSampler.mSamplingSpec.mScaling, std::array<float, 2>{ 3.0f, 4.0f } ) );
        }
    }
}