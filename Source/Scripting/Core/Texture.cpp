#include "Texture.h"
#include "Vector.h"

#include "Core/Memory.h"
#include "Core/TextureData.h"
#include "Cuda/MultiTensor.h"

#include "TensorOps/Scope.h"

#include "Core/Logging.h"

namespace LTSE::Core
{
    using namespace sol;

    TextureData::sCreateInfo ParseCreateInfo( sol::table aTable )
    {
        TextureData::sCreateInfo lCreateInfo{};
        lCreateInfo.mType      = aTable["type"].valid() ? aTable["type"] : eTextureType::TEXTURE_2D;
        lCreateInfo.mFormat    = aTable["color_format"].valid() ? aTable["color_format"] : eColorFormat::UNDEFINED;
        lCreateInfo.mWidth     = aTable["width"].valid() ? aTable["width"] : 0;
        lCreateInfo.mHeight    = aTable["height"].valid() ? aTable["height"] : 0;
        lCreateInfo.mDepth     = aTable["depth"].valid() ? aTable["depth"] : 0;
        lCreateInfo.mMipLevels = aTable["mip_levels"].valid() ? aTable["mip_levels"] : 0;

        return lCreateInfo;
    }

    sImageData ParseImageData( sol::table aTable )
    {
        sImageData lImageData{};
        lImageData.mFormat = aTable["color_format"].valid() ? aTable["color_format"] : eColorFormat::UNDEFINED;
        lImageData.mWidth  = aTable["width"].valid() ? aTable["width"] : 0;
        lImageData.mHeight = aTable["height"].valid() ? aTable["height"] : 0;

        auto lPixelData       = aTable.get<sol::table>( "pixel_data" );
        auto lPixelDataSize   = lPixelData.size();
        lImageData.mByteSize  = lPixelDataSize;
        lImageData.mPixelData = (uint8_t *)malloc( lPixelDataSize );

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
            auto lDataPointer = lPixelData.as<std::vector<uint8_t>>();
            for( uint32_t i = 0; i < lPixelDataSize; i++ )
            {
                auto lPx                 = lDataPointer[i];
                lImageData.mPixelData[i] = lPx;
            }
        }

        return lImageData;
    }

    sTextureSamplingInfo ParseSamplerInfo( sol::table aTable )
    {
        sTextureSamplingInfo lSamplingInfo{};
        lSamplingInfo.mMinification  = aTable["minification"].valid() ? aTable["minification"] : eSamplerFilter::LINEAR;
        lSamplingInfo.mMagnification = aTable["magnification"].valid() ? aTable["magnification"] : eSamplerFilter::LINEAR;
        lSamplingInfo.mMip           = aTable["mip"].valid() ? aTable["mip"] : eSamplerMipmap::LINEAR;
        lSamplingInfo.mWrapping      = aTable["wrapping"].valid() ? aTable["wrapping"] : eSamplerWrapping::CLAMP_TO_BORDER;

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
        aScriptingState.new_enum( "eTextureType", "TEXTURE_2D", eTextureType::TEXTURE_2D, "TEXTURE_3D", eTextureType::TEXTURE_3D );

        // clang-format off
        aScriptingState.new_enum( "eColorFormat",
            "UNDEFINED",           eColorFormat::UNDEFINED,
            "R32_FLOAT",           eColorFormat::R32_FLOAT,
            "RG32_FLOAT",          eColorFormat::RG32_FLOAT,
            "RGB32_FLOAT",         eColorFormat::RGB32_FLOAT,
            "RGBA32_FLOAT",        eColorFormat::RGBA32_FLOAT,
            "R16_FLOAT",           eColorFormat::R16_FLOAT,
            "RG16_FLOAT",          eColorFormat::RG16_FLOAT,
            "RGB16_FLOAT",         eColorFormat::RGB16_FLOAT,
            "RGBA16_FLOAT",        eColorFormat::RGBA16_FLOAT,
            "R8_UNORM",            eColorFormat::R8_UNORM,
            "RG8_UNORM",           eColorFormat::RG8_UNORM,
            "RGB8_UNORM",          eColorFormat::RGB8_UNORM,
            "RGBA8_UNORM",         eColorFormat::RGBA8_UNORM,
            "D16_UNORM",           eColorFormat::D16_UNORM,
            "X8_D24_UNORM_PACK32", eColorFormat::X8_D24_UNORM_PACK32,
            "D32_SFLOAT",          eColorFormat::D32_SFLOAT,
            "S8_UINT",             eColorFormat::S8_UINT,
            "D16_UNORM_S8_UINT",   eColorFormat::D16_UNORM_S8_UINT,
            "D24_UNORM_S8_UINT",   eColorFormat::D24_UNORM_S8_UINT,
            "D32_UNORM_S8_UINT",   eColorFormat::D32_UNORM_S8_UINT,
            "BGR8_SRGB",           eColorFormat::BGR8_SRGB,
            "BGRA8_SRGB",          eColorFormat::BGRA8_SRGB );
        // clang-format on

        aScriptingState.new_enum( "eSamplerFilter", "NEAREST", eSamplerFilter::NEAREST, "LINEAR", eSamplerFilter::LINEAR );
        aScriptingState.new_enum( "eSamplerMipmap", "NEAREST", eSamplerMipmap::NEAREST, "LINEAR", eSamplerMipmap::LINEAR );

        // clang-format off
        aScriptingState.new_enum( "eSamplerWrapping",
            "REPEAT",          eSamplerWrapping::REPEAT,
            "MIRRORED_REPEAT", eSamplerWrapping::MIRRORED_REPEAT,
            "CLAMP_TO_EDGE",   eSamplerWrapping::CLAMP_TO_EDGE,
            "CLAMP_TO_BORDER", eSamplerWrapping::CLAMP_TO_BORDER,
            "MIRROR_CLAMP_TO_BORDER", eSamplerWrapping::MIRROR_CLAMP_TO_BORDER );
        // clang-format on

        auto lTextureData2DType = aScriptingState.new_usertype<TextureData2D>( "TextureData2D" );

        // clang-format off
        lTextureData2DType[call_constructor] = factories(
            []( sol::table aCreateInfo ) { return TextureData2D( ParseCreateInfo(aCreateInfo) ); },
            []( sol::table aCreateInfo, sol::table aImageData ){ return TextureData2D( ParseCreateInfo( aCreateInfo ), ParseImageData( aImageData ) ); },
            []( std::string const &aImagePath )
            {
                TextureData::sCreateInfo lCreateInfo{};
                return TextureData2D( lCreateInfo, fs::path(aImagePath) );
            }
        );
        // clang-format on

        lTextureData2DType["get_image_data"] = [&]( TextureData2D &aSelf, sol::this_state aScriptState )
        {
            sol::table lDataTable( aScriptState, sol::new_table{} );
            auto lImageData = aSelf.GetImageData();

            lDataTable["color_format"] = lImageData.mFormat;
            lDataTable["width"]        = lImageData.mWidth;
            lDataTable["height"]       = lImageData.mHeight;

            auto lDataVector = std::vector<uint8_t>( lImageData.mByteSize );
            for( uint32_t i = 0; i < lImageData.mByteSize; i++ )
                lDataVector[i] = lImageData.mPixelData[i];
            lDataTable["pixel_data"] = lDataVector;

            return lDataTable;
        };

        aScriptingState["load_image"] = []( std::string aPath, sol::this_state aScriptState )
        {
            auto lImageData = LoadImageData( fs::path( aPath ) );

            sol::table lDataTable( aScriptState, sol::new_table{} );

            lDataTable["color_format"] = lImageData.mFormat;
            lDataTable["width"]        = lImageData.mWidth;
            lDataTable["height"]       = lImageData.mHeight;
            auto lDataVector           = std::vector<uint8_t>( lImageData.mByteSize );
            for( uint32_t i = 0; i < lImageData.mByteSize; i++ )
                lDataVector[i] = lImageData.mPixelData[i];
            lDataTable["pixel_data"] = lDataVector;

            return lDataTable;
        };

        auto lTextureSampler2DType = aScriptingState.new_usertype<TextureSampler2D>( "TextureSampler2D" );

        lTextureSampler2DType[call_constructor] =
            factories( []( TextureData2D const &aTexture, sol::table aCreateInfo ) { return TextureSampler2D( aTexture, ParseSamplerInfo( aCreateInfo ) ); } );

        lTextureSampler2DType["fetch"] = []( TextureSampler2D &aSelf, float x, float y ) { return aSelf.Fetch( x, y ); };
    }

    void OpenCoreLibrary( sol::table &aScriptingState )
    {
        RequireTexture( aScriptingState );
        OpenVectorLibrary( aScriptingState );
    }
}; // namespace LTSE::Core