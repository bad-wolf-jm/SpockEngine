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

    texture_create_info_t ParseCreateInfo( sol::table aTable )
    {
        texture_create_info_t lCreateInfo{};
        lCreateInfo.mType      = aTable["type"].valid() ? aTable["type"] : texture_type_t::TEXTURE_2D;
        lCreateInfo.mFormat    = aTable["color_format"].valid() ? aTable["color_format"] : color_format_t::UNDEFINED;
        lCreateInfo.mWidth     = aTable["width"].valid() ? aTable["width"] : 0;
        lCreateInfo.mHeight    = aTable["height"].valid() ? aTable["height"] : 0;
        lCreateInfo.mDepth     = aTable["depth"].valid() ? aTable["depth"] : 0;
        lCreateInfo.mMipLevels = aTable["mip_levels"].valid() ? aTable["mip_levels"] : 0;

        return lCreateInfo;
    }

    image_data_t ParseImageData( sol::table aTable )
    {
        image_data_t lImageData{};
        lImageData.mFormat = aTable["color_format"].valid() ? aTable["color_format"] : color_format_t::UNDEFINED;
        lImageData.mWidth  = aTable["width"].valid() ? aTable["width"] : 0;
        lImageData.mHeight = aTable["height"].valid() ? aTable["height"] : 0;

        auto lPixelData       = aTable.get<sol::table>( "pixel_data" );
        auto lPixelDataSize   = lPixelData.size();
        lImageData.mByteSize  = lPixelDataSize;
        lImageData.mPixelData = vector_t<uint8_t>( lPixelDataSize );

        if( lPixelData.get_type() == sol::type::table )
        {
            for( uint32_t i = 0; i < lPixelDataSize; i++ )
            {
                auto lPx                 = lPixelData.get<uint8_t>( i + 1 );
                lImageData.mPixelData[i] = lPx;
            }
        }
        else
        {
            auto lDataPointer = lPixelData.as<vector_t<uint8_t>>();
            for( uint32_t i = 0; i < lPixelDataSize; i++ )
            {
                auto lPx                 = lDataPointer[i];
                lImageData.mPixelData[i] = lPx;
            }
        }

        return lImageData;
    }

    texture_sampling_info_t ParseSamplerInfo( sol::table aTable )
    {
        texture_sampling_info_t lSamplingInfo{};
        lSamplingInfo.mFilter = aTable["minification"].valid() ? aTable["minification"] : sampler_filter_t::LINEAR;
        // lSamplingInfo.mMagnification = aTable["magnification"].valid() ? aTable["magnification"] : eSamplerFilter::LINEAR;
        lSamplingInfo.mMipFilter = aTable["mip"].valid() ? aTable["mip"] : sampler_mipmap_t::LINEAR;
        lSamplingInfo.mWrapping  = aTable["wrapping"].valid() ? aTable["wrapping"] : sampler_wrapping_t::CLAMP_TO_BORDER;

        auto lScaling = aTable.get<sol::table>( "scaling" );
        if( !lScaling.valid() )
        {
            lSamplingInfo.mScaling = { 1.0f, 1.0f };
        }
        else
        {
            lSamplingInfo.mScaling[0] = lScaling["x"].valid() ? lScaling["x"] : 1.0f;
            lSamplingInfo.mScaling[1] = lScaling["y"].valid() ? lScaling["y"] : 1.0f;
        }
        auto lOffset = aTable.get<sol::table>( "offset" );
        if( !lOffset.valid() )
        {
            lSamplingInfo.mOffset = { 0.0f, 0.0f };
        }
        else
        {
            lSamplingInfo.mOffset[0] = lOffset["x"].valid() ? lOffset["x"] : 0.0f;
            lSamplingInfo.mOffset[1] = lOffset["y"].valid() ? lOffset["y"] : 0.0f;
        }
        auto lBorderColor = aTable.get<sol::table>( "border_color" );
        if( !lBorderColor.valid() )
        {
            lSamplingInfo.mBorderColor = { 0.0f, 0.0f, 0.0f, 0.0f };
        }
        else
        {
            lSamplingInfo.mBorderColor[0] = lBorderColor["r"].valid() ? lBorderColor["r"] : 0.0f;
            lSamplingInfo.mBorderColor[1] = lBorderColor["g"].valid() ? lBorderColor["g"] : 0.0f;
            lSamplingInfo.mBorderColor[2] = lBorderColor["b"].valid() ? lBorderColor["b"] : 0.0f;
            lSamplingInfo.mBorderColor[3] = lBorderColor["a"].valid() ? lBorderColor["a"] : 0.0f;
        }

        return lSamplingInfo;
    }

    void RequireTexture( sol::table &aScriptingState )
    {
        aScriptingState.new_enum( "eTextureType", "TEXTURE_2D", texture_type_t::TEXTURE_2D, "TEXTURE_3D", texture_type_t::TEXTURE_3D );

        // clang-format off
        aScriptingState.new_enum( "eColorFormat",
            "UNDEFINED",           color_format_t::UNDEFINED,
            "R32_FLOAT",           color_format_t::R32_FLOAT,
            "RG32_FLOAT",          color_format_t::RG32_FLOAT,
            "RGB32_FLOAT",         color_format_t::RGB32_FLOAT,
            "RGBA32_FLOAT",        color_format_t::RGBA32_FLOAT,
            "R16_FLOAT",           color_format_t::R16_FLOAT,
            "RG16_FLOAT",          color_format_t::RG16_FLOAT,
            "RGB16_FLOAT",         color_format_t::RGB16_FLOAT,
            "RGBA16_FLOAT",        color_format_t::RGBA16_FLOAT,
            "R8_UNORM",            color_format_t::R8_UNORM,
            "RG8_UNORM",           color_format_t::RG8_UNORM,
            "RGB8_UNORM",          color_format_t::RGB8_UNORM,
            "RGBA8_UNORM",         color_format_t::RGBA8_UNORM,
            "D16_UNORM",           color_format_t::D16_UNORM,
            "X8_D24_UNORM_PACK32", color_format_t::X8_D24_UNORM_PACK32,
            "D32_SFLOAT",          color_format_t::D32_SFLOAT,
            "S8_UINT",             color_format_t::S8_UINT,
            "D16_UNORM_S8_UINT",   color_format_t::D16_UNORM_S8_UINT,
            "D24_UNORM_S8_UINT",   color_format_t::D24_UNORM_S8_UINT,
            "D32_UNORM_S8_UINT",   color_format_t::D32_UNORM_S8_UINT,
            "BGR8_SRGB",           color_format_t::BGR8_SRGB,
            "BGRA8_SRGB",          color_format_t::BGRA8_SRGB );
        // clang-format on

        aScriptingState.new_enum( "eSamplerFilter", "NEAREST", sampler_filter_t::NEAREST, "LINEAR", sampler_filter_t::LINEAR );
        aScriptingState.new_enum( "eSamplerMipmap", "NEAREST", sampler_mipmap_t::NEAREST, "LINEAR", sampler_mipmap_t::LINEAR );

        // clang-format off
        aScriptingState.new_enum( "eSamplerWrapping",
            "REPEAT",          sampler_wrapping_t::REPEAT,
            "MIRRORED_REPEAT", sampler_wrapping_t::MIRRORED_REPEAT,
            "CLAMP_TO_EDGE",   sampler_wrapping_t::CLAMP_TO_EDGE,
            "CLAMP_TO_BORDER", sampler_wrapping_t::CLAMP_TO_BORDER,
            "MIRROR_CLAMP_TO_BORDER", sampler_wrapping_t::MIRROR_CLAMP_TO_BORDER );
        // clang-format on

        // auto lTextureData2DType = aScriptingState.new_usertype<texture_data2d_t>( "TextureData2D" );

        // clang-format off
        // lTextureData2DType[call_constructor] = factories(
        //     []( sol::table aCreateInfo ) { return texture_data2d_t( ParseCreateInfo(aCreateInfo) ); },
        //     []( sol::table aCreateInfo, sol::table aImageData ){ return texture_data2d_t( ParseCreateInfo( aCreateInfo ), ParseImageData( aImageData ) ); },
        //     []( std::string const &aImagePath )
        //     {
        //         texture_create_info_t lCreateInfo{};
        //         return texture_data2d_t( lCreateInfo, fs::path(aImagePath) );
        //     }
        // );
        // clang-format on

        // lTextureData2DType["get_image_data"] = [&]( texture_data2d_t &aSelf, sol::this_state aScriptState )
        // {
        //     sol::table lDataTable( aScriptState, sol::new_table{} );
        //     auto       lImageData = aSelf.GetImageData();

        //     lDataTable["color_format"] = lImageData.mFormat;
        //     lDataTable["width"]        = lImageData.mWidth;
        //     lDataTable["height"]       = lImageData.mHeight;

        //     auto lDataVector = vector_t<uint8_t>( lImageData.mByteSize );
        //     for( uint32_t i = 0; i < lImageData.mByteSize; i++ )
        //         lDataVector[i] = lImageData.mPixelData[i];
        //     lDataTable["pixel_data"] = lDataVector;

        //     return lDataTable;
        // };

        aScriptingState["load_image"] = []( std::string aPath, sol::this_state aScriptState )
        {
            auto lImageData = LoadImageData( fs::path( aPath ) );

            sol::table lDataTable( aScriptState, sol::new_table{} );

            lDataTable["color_format"] = lImageData.mFormat;
            lDataTable["width"]        = lImageData.mWidth;
            lDataTable["height"]       = lImageData.mHeight;
            auto lDataVector           = vector_t<uint8_t>( lImageData.mByteSize );
            for( uint32_t i = 0; i < lImageData.mByteSize; i++ )
                lDataVector[i] = lImageData.mPixelData[i];
            lDataTable["pixel_data"] = lDataVector;

            return lDataTable;
        };

        // auto lTextureSampler2DType = aScriptingState.new_usertype<texture_data_sampler2d_t>( "TextureSampler2D" );

        // lTextureSampler2DType[call_constructor] =
        //     factories( []( texture_data2d_t const &aTexture, sol::table aCreateInfo )
        //                { return texture_data_sampler2d_t( aTexture, ParseSamplerInfo( aCreateInfo ) ); } );

        // lTextureSampler2DType["fetch"] = []( TextureSampler2D &aSelf, float x, float y ) { return aSelf.Fetch( x, y ); };
    }

    void OpenCoreLibrary( sol::table &aScriptingState )
    {
        RequireTexture( aScriptingState );
        OpenVectorLibrary( aScriptingState );
    }
}; // namespace SE::Core