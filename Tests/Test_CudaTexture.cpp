#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "TestUtils.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"

#include "Core/CUDA/Texture/Texture2D.h"

namespace fs = std::filesystem;

using namespace SE;
using namespace SE::Core;
using namespace SE::Cuda;

TEST_CASE( "Loading Cuda 2D textures", "[CORE_CUDA_TEXTURES]" )
{
    fs::path lTestDataRoot( "C:\\GitLab\\SpockEngine\\Tests\\Data" );

    SECTION( "Load 2D textures from data" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                        0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                        0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            std::vector<uint8_t> lPixelDataVector( 16 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ),
                         16 * sizeof( uint32_t ) );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = color_format_t::RGBA8_UNORM;
            lTextureCreateInfo.mWidth  = 4;
            lTextureCreateInfo.mHeight = 4;
            texture2d_t lTexture( lTextureCreateInfo, lPixelDataVector );
            REQUIRE( lTexture.mSpec.mFormat == color_format_t::RGBA8_UNORM );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };
            std::vector<uint8_t> lPixelDataVector( 16 * sizeof( float ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ),
                         16 * sizeof( float ) );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = color_format_t::R32_FLOAT;
            lTextureCreateInfo.mWidth  = 4;
            lTextureCreateInfo.mHeight = 4;
            texture2d_t lTexture( lTextureCreateInfo, lPixelDataVector );
            REQUIRE( lTexture.mSpec.mFormat == color_format_t::R32_FLOAT );
        }
    }

    SECTION( "Create cuda texture sImageData" )
    {
        {
            uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
                                        0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                        0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format_t::RGBA8_UNORM;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lCudaTextureCreateInfo{};
            texture2d_t          lCudaTexture( lCudaTextureCreateInfo, lImageDataStruct );
            REQUIRE( lCudaTexture.mSpec.mFormat == color_format_t::RGBA8_UNORM );
        }

        {
            float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

            image_data_t lImageDataStruct{};
            lImageDataStruct.mFormat    = color_format_t::R32_FLOAT;
            lImageDataStruct.mWidth     = 4;
            lImageDataStruct.mHeight    = 4;
            lImageDataStruct.mByteSize  = 16 * sizeof( float );
            lImageDataStruct.mPixelData = std::vector<uint8_t>(
                reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

            texture_create_info_t lCudaTextureCreateInfo{};
            texture2d_t          lCudaTexture( lCudaTextureCreateInfo, lImageDataStruct );
            REQUIRE( lCudaTexture.mSpec.mFormat == color_format_t::R32_FLOAT );
        }
    }

    // SECTION( "Load cuda 2D textures from data" )
    // {
    //     {
    //         uint32_t lImageData[16] = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
    //                                     0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
    //                                     0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 };

    //         image_data_t lImageDataStruct{};
    //         lImageDataStruct.mFormat    = color_format_t::RGBA8_UNORM;
    //         lImageDataStruct.mWidth     = 4;
    //         lImageDataStruct.mHeight    = 4;
    //         lImageDataStruct.mByteSize  = 16 * sizeof( uint32_t );
    //         lImageDataStruct.mPixelData = std::vector<uint8_t>(
    //             reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

    //         texture_create_info_t lTextureCreateInfo{};
    //         lTextureCreateInfo.mMipLevels = 1;
    //         texture_data2d_t lTexture( lTextureCreateInfo, lImageDataStruct );

    //         texture_create_info_t lCudaTextureCreateInfo{};
    //         texture2d_t          lCudaTexture( lCudaTextureCreateInfo, lTexture.GetImageData() );
    //         REQUIRE( lCudaTexture.mSpec.mFormat == color_format_t::RGBA8_UNORM );
    //     }

    //     {
    //         float lImageData[16] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };

    //         image_data_t lImageDataStruct{};
    //         lImageDataStruct.mFormat    = color_format_t::R32_FLOAT;
    //         lImageDataStruct.mWidth     = 4;
    //         lImageDataStruct.mHeight    = 4;
    //         lImageDataStruct.mByteSize  = 16 * sizeof( float );
    //         lImageDataStruct.mPixelData = std::vector<uint8_t>(
    //             reinterpret_cast<uint8_t *>( lImageData ), reinterpret_cast<uint8_t *>( lImageData ) + lImageDataStruct.mByteSize );

    //         texture_create_info_t lTextureCreateInfo{};
    //         lTextureCreateInfo.mMipLevels = 1;
    //         texture_data2d_t      lTexture( lTextureCreateInfo, lImageDataStruct );
    //         texture_create_info_t lCudaTextureCreateInfo{};
    //         texture2d_t          lCudaTexture( lCudaTextureCreateInfo, lTexture.GetImageData() );
    //         REQUIRE( lCudaTexture.mSpec.mFormat == color_format_t::R32_FLOAT );
    //     }
    // }

    SECTION( "Sampling 2D textures" )
    {
        {
            uint32_t lImageData[9] = { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                       0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF };

            std::vector<uint8_t> lPixelDataVector( 9 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ),
                         9 * sizeof( uint32_t ) );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = color_format_t::RGBA8_UNORM;
            lTextureCreateInfo.mWidth  = 3;
            lTextureCreateInfo.mHeight = 3;
            ref_t<texture2d_t> lTexture  = New<texture2d_t>( lTextureCreateInfo, lPixelDataVector );

            texture_sampling_info_t lSamplingInfo{};
            lSamplingInfo.mNormalizedValues        = true;
            Cuda::texture_sampler2d_t lTextureSampler = Cuda::texture_sampler2d_t( lTexture, lSamplingInfo );

            REQUIRE( lTextureSampler.mSpec.mFilter == sampler_filter_t::LINEAR );
            REQUIRE( lTextureSampler.mSpec.mWrapping == sampler_wrapping_t::CLAMP_TO_BORDER );
            REQUIRE( lTextureSampler.mSpec.mBorderColor[0] == 0.0f );
            REQUIRE( lTextureSampler.mSpec.mBorderColor[1] == 0.0f );
            REQUIRE( lTextureSampler.mSpec.mBorderColor[2] == 0.0f );
            REQUIRE( lTextureSampler.mSpec.mBorderColor[3] == 0.0f );

            REQUIRE( TestUtils::VectorEqual( lTextureSampler.mSpec.mScaling, std::array<float, 2>{ 1.0f, 1.0f } ) );
        }

        {
            uint32_t lImageData[9] = { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                       0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF };

            std::vector<uint8_t> lPixelDataVector( 9 * sizeof( uint32_t ) );
            std::memcpy( reinterpret_cast<void *>( lPixelDataVector.data() ), reinterpret_cast<void *>( lImageData ),
                         9 * sizeof( uint32_t ) );

            texture_create_info_t lTextureCreateInfo{};
            lTextureCreateInfo.mFormat = color_format_t::RGBA8_UNORM;
            lTextureCreateInfo.mWidth  = 3;
            lTextureCreateInfo.mHeight = 3;
            ref_t<texture2d_t> lTexture  = New<texture2d_t>( lTextureCreateInfo, lPixelDataVector );

            texture_sampling_info_t lSamplingInfo{};
            lSamplingInfo.mScaling                 = std::array<float, 2>{ 3.0f, 4.0f };
            lSamplingInfo.mNormalizedValues        = true;
            Cuda::texture_sampler2d_t lTextureSampler = Cuda::texture_sampler2d_t( lTexture, lSamplingInfo );

            REQUIRE( TestUtils::VectorEqual( lTextureSampler.mSpec.mScaling, std::array<float, 2>{ 3.0f, 4.0f } ) );
        }
    }
}