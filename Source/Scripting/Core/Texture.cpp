#include "Texture.h"
#include "Vector.h"

#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Definitions.h"

#include "TensorOps/Scope.h"

#include "Core/Logging.h"

namespace SE::Core
{
    using namespace sol;

    texture_create_info_t parse_create_info( sol::table table )
    {
        texture_create_info_t createInfo{};
        createInfo.mType      = table["type"].valid() ? table["type"] : texture_type_t::TEXTURE_2D;
        createInfo.mFormat    = table["color_format"].valid() ? table["color_format"] : color_format::UNDEFINED;
        createInfo.mWidth     = table["width"].valid() ? table["width"] : 0;
        createInfo.mHeight    = table["height"].valid() ? table["height"] : 0;
        createInfo.mDepth     = table["depth"].valid() ? table["depth"] : 0;
        createInfo.mMipLevels = table["mip_levels"].valid() ? table["mip_levels"] : 0;

        return createInfo;
    }

    image_data_t parse_image_data( sol::table table )
    {
        image_data_t imageData{};
        imageData.mFormat = table["color_format"].valid() ? table["color_format"] : color_format::UNDEFINED;
        imageData.mWidth  = table["width"].valid() ? table["width"] : 0;
        imageData.mHeight = table["height"].valid() ? table["height"] : 0;

        auto pixelData       = table.get<sol::table>( "pixel_data" );
        auto pixelDataSize   = pixelData.size();
        imageData.mByteSize  = pixelDataSize;
        imageData.mPixelData = vector_t<uint8_t>( pixelDataSize );

        if( pixelData.get_type() == sol::type::table )
        {
            for( uint32_t i = 0; i < pixelDataSize; i++ )
            {
                imageData.mPixelData[i] = pixelData.get<uint8_t>( i + 1 );
            }
        }
        else
        {
            auto dataPointer = pixelData.as<vector_t<uint8_t>>();
            for( uint32_t i = 0; i < pixelDataSize; i++ )
            {
                imageData.mPixelData[i] = dataPointer[i];
            }
        }

        return imageData;
    }

    texture_sampling_info_t parse_sampler_info( sol::table table )
    {
        texture_sampling_info_t samplingInfo{};
        samplingInfo.mFilter = table["minification"].valid() ? table["minification"] : sampler_filter_t::LINEAR;
        // samplingInfo.mMagnification = table["magnification"].valid() ? table["magnification"] : eSamplerFilter::LINEAR;
        samplingInfo.mMipFilter = table["mip"].valid() ? table["mip"] : sampler_mipmap_t::LINEAR;
        samplingInfo.mWrapping  = table["wrapping"].valid() ? table["wrapping"] : sampler_wrapping_t::CLAMP_TO_BORDER;

        auto lScaling = table.get<sol::table>( "scaling" );
        if( !lScaling.valid() )
        {
            samplingInfo.mScaling = { 1.0f, 1.0f };
        }
        else
        {
            samplingInfo.mScaling[0] = lScaling["x"].valid() ? lScaling["x"] : 1.0f;
            samplingInfo.mScaling[1] = lScaling["y"].valid() ? lScaling["y"] : 1.0f;
        }
        auto offset = table.get<sol::table>( "offset" );
        if( !offset.valid() )
        {
            samplingInfo.mOffset = { 0.0f, 0.0f };
        }
        else
        {
            samplingInfo.mOffset[0] = offset["x"].valid() ? offset["x"] : 0.0f;
            samplingInfo.mOffset[1] = offset["y"].valid() ? offset["y"] : 0.0f;
        }
        auto borderColor = table.get<sol::table>( "border_color" );
        if( !borderColor.valid() )
        {
            samplingInfo.mBorderColor = { 0.0f, 0.0f, 0.0f, 0.0f };
        }
        else
        {
            samplingInfo.mBorderColor[0] = borderColor["r"].valid() ? borderColor["r"] : 0.0f;
            samplingInfo.mBorderColor[1] = borderColor["g"].valid() ? borderColor["g"] : 0.0f;
            samplingInfo.mBorderColor[2] = borderColor["b"].valid() ? borderColor["b"] : 0.0f;
            samplingInfo.mBorderColor[3] = borderColor["a"].valid() ? borderColor["a"] : 0.0f;
        }

        return samplingInfo;
    }

    void require_texture( sol::table &scriptingState )
    {
        scriptingState.new_enum( "eTextureType", "TEXTURE_2D", texture_type_t::TEXTURE_2D, "TEXTURE_3D", texture_type_t::TEXTURE_3D );

        // clang-format off
        scriptingState.new_enum( "eColorFormat",
            "UNDEFINED",           color_format::UNDEFINED,
            "R32_FLOAT",           color_format::R32_FLOAT,
            "RG32_FLOAT",          color_format::RG32_FLOAT,
            "RGB32_FLOAT",         color_format::RGB32_FLOAT,
            "RGBA32_FLOAT",        color_format::RGBA32_FLOAT,
            "R16_FLOAT",           color_format::R16_FLOAT,
            "RG16_FLOAT",          color_format::RG16_FLOAT,
            "RGB16_FLOAT",         color_format::RGB16_FLOAT,
            "RGBA16_FLOAT",        color_format::RGBA16_FLOAT,
            "R8_UNORM",            color_format::R8_UNORM,
            "RG8_UNORM",           color_format::RG8_UNORM,
            "RGB8_UNORM",          color_format::RGB8_UNORM,
            "RGBA8_UNORM",         color_format::RGBA8_UNORM,
            "D16_UNORM",           color_format::D16_UNORM,
            "X8_D24_UNORM_PACK32", color_format::X8_D24_UNORM_PACK32,
            "D32_SFLOAT",          color_format::D32_SFLOAT,
            "S8_UINT",             color_format::S8_UINT,
            "D16_UNORM_S8_UINT",   color_format::D16_UNORM_S8_UINT,
            "D24_UNORM_S8_UINT",   color_format::D24_UNORM_S8_UINT,
            "D32_UNORM_S8_UINT",   color_format::D32_UNORM_S8_UINT,
            "BGR8_SRGB",           color_format::BGR8_SRGB,
            "BGRA8_SRGB",          color_format::BGRA8_SRGB );
        // clang-format on

        scriptingState.new_enum( "eSamplerFilter", "NEAREST", sampler_filter_t::NEAREST, "LINEAR", sampler_filter_t::LINEAR );
        scriptingState.new_enum( "eSamplerMipmap", "NEAREST", sampler_mipmap_t::NEAREST, "LINEAR", sampler_mipmap_t::LINEAR );

        // clang-format off
        scriptingState.new_enum( "eSamplerWrapping",
            "REPEAT",          sampler_wrapping_t::REPEAT,
            "MIRRORED_REPEAT", sampler_wrapping_t::MIRRORED_REPEAT,
            "CLAMP_TO_EDGE",   sampler_wrapping_t::CLAMP_TO_EDGE,
            "CLAMP_TO_BORDER", sampler_wrapping_t::CLAMP_TO_BORDER,
            "MIRROR_CLAMP_TO_BORDER", sampler_wrapping_t::MIRROR_CLAMP_TO_BORDER );
        // clang-format on

        // auto lTextureData2DType = scriptingState.new_usertype<texture_data2d_t>( "TextureData2D" );

        // clang-format off
        // lTextureData2DType[call_constructor] = factories(
        //     []( sol::table aCreateInfo ) { return texture_data2d_t( parse_create_info(aCreateInfo) ); },
        //     []( sol::table aCreateInfo, sol::table aImageData ){ return texture_data2d_t( parse_create_info( aCreateInfo ), parse_image_data( aImageData ) ); },
        //     []( std::string const &aImagePath )
        //     {
        //         texture_create_info_t createInfo{};
        //         return texture_data2d_t( createInfo, fs::path(aImagePath) );
        //     }
        // );
        // clang-format on

        // lTextureData2DType["get_image_data"] = [&]( texture_data2d_t &aSelf, sol::this_state scriptState )
        // {
        //     sol::table dataTable( scriptState, sol::new_table{} );
        //     auto       imageData = aSelf.GetImageData();

        //     dataTable["color_format"] = imageData.mFormat;
        //     dataTable["width"]        = imageData.mWidth;
        //     dataTable["height"]       = imageData.mHeight;

        //     auto lDataVector = vector_t<uint8_t>( imageData.mByteSize );
        //     for( uint32_t i = 0; i < imageData.mByteSize; i++ )
        //         lDataVector[i] = imageData.mPixelData[i];
        //     dataTable["pixel_data"] = lDataVector;

        //     return dataTable;
        // };

        scriptingState["load_image"] = []( std::string path, sol::this_state scriptState )
        {
            auto imageData = LoadImageData( fs::path( path ) );

            sol::table dataTable( scriptState, sol::new_table{} );

            dataTable["color_format"] = imageData.mFormat;
            dataTable["width"]        = imageData.mWidth;
            dataTable["height"]       = imageData.mHeight;
            auto lDataVector           = vector_t<uint8_t>( imageData.mByteSize );
            for( uint32_t i = 0; i < imageData.mByteSize; i++ )
                lDataVector[i] = imageData.mPixelData[i];
            dataTable["pixel_data"] = lDataVector;

            return dataTable;
        };

        // auto lTextureSampler2DType = scriptingState.new_usertype<texture_data_sampler2d_t>( "TextureSampler2D" );

        // lTextureSampler2DType[call_constructor] =
        //     factories( []( texture_data2d_t const &aTexture, sol::table aCreateInfo )
        //                { return texture_data_sampler2d_t( aTexture, parse_sampler_info( aCreateInfo ) ); } );

        // lTextureSampler2DType["fetch"] = []( TextureSampler2D &aSelf, float x, float y ) { return aSelf.Fetch( x, y ); };
    }

    void open_core_library( sol::table &scriptingState )
    {
        require_texture( scriptingState );
        open_vector_library( scriptingState );
    }
}; // namespace SE::Core