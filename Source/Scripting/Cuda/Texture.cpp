#include "Texture.h"

#include "Core/Memory.h"
#include "Core/TextureData.h"
#include "Cuda/Texture2D.h"

#include "TensorOps/Scope.h"

#include "Core/Logging.h"

#include "Scripting/Core/Texture.h"

namespace LTSE::Core
{
    using namespace sol;

    Cuda::sTextureCreateInfo ParseCudaCreateInfo( sol::table aTable )
    {
        Cuda::sTextureCreateInfo lCreateInfo{};
        lCreateInfo.mFilterMode            = aTable["filter_mode"].valid() ? aTable["filter_mode"] : eSamplerFilter::LINEAR;
        lCreateInfo.mWrappingMode          = aTable["wrapping"].valid() ? aTable["wrapping"] : eSamplerWrapping::CLAMP_TO_BORDER;
        lCreateInfo.mFormat                = aTable["color_format"].valid() ? aTable["color_format"] : eColorFormat::UNDEFINED;
        lCreateInfo.mWidth                 = aTable["width"].valid() ? aTable["width"] : 0;
        lCreateInfo.mHeight                = aTable["height"].valid() ? aTable["height"] : 0;
        lCreateInfo.mNormalizedCoordinates = aTable["normalized_coordinates"].valid() ? aTable["normalized_coordinates"] : false;
        lCreateInfo.mNormalizedValues      = aTable["normalized_values"].valid() ? aTable["normalized_values"] : false;

        return lCreateInfo;
    }

    void RequireCudaTexture( sol::table &aScriptingState )
    {
        auto lTextureData2DType = aScriptingState.new_usertype<Cuda::Texture2D>( "Texture2D" );

        // clang-format off
        lTextureData2DType[call_constructor] = factories(
            []( sol::table aCreateInfo, sol::table aImageData ) {
                return New<Cuda::Texture2D>( ParseCudaCreateInfo( aCreateInfo ), ParseImageData( aImageData ) );
            },
            []( sol::table aCreateInfo, std::vector<uint8_t> aImageData ) {
                return New<Cuda::Texture2D>( ParseCudaCreateInfo( aCreateInfo ), aImageData );
            }
        );
        // clang-format on

        auto lTextureSampler2DType = aScriptingState.new_usertype<Cuda::TextureSampler2D>( "TextureSampler2D" );

        lTextureSampler2DType[call_constructor] =
            factories( []( Ref<Cuda::Texture2D> &aTexture, sol::table aCreateInfo ) { return Cuda::TextureSampler2D( aTexture, ParseSamplerInfo( aCreateInfo ) ); } );
        lTextureSampler2DType["spec"] = &Cuda::TextureSampler2D::mSamplingSpec;
        lTextureSampler2DType["device_data"] = &Cuda::TextureSampler2D::mDeviceData;

        aScriptingState["load_texture_sampler"] = overload(
            []( sol::table aCreateInfo, std::string const &aTexturePath )
            {
                TextureData2D::sCreateInfo lCreateInfo{};
                TextureData2D lTextureData( lCreateInfo, aTexturePath );

                LTSE::Cuda::sTextureCreateInfo lTextureCreateInfo = ParseCudaCreateInfo( aCreateInfo );

                switch( lTextureData.mSpec.mFormat )
                {
                case LTSE::Core::eColorFormat::R32_FLOAT:
                    lTextureCreateInfo.mNormalizedValues = false;
                    break;
                default:
                    lTextureCreateInfo.mNormalizedValues = true;
                    break;
                }

                Ref<LTSE::Cuda::Texture2D> lTexture = New<LTSE::Cuda::Texture2D>( lTextureCreateInfo, lTextureData.GetImageData() );

                sTextureSamplingInfo lSamplingInfo{};
                lSamplingInfo.mScaling       = std::array<float, 2>{ 1.0f, 1.0f };
                lSamplingInfo.mMinification  = LTSE::Core::eSamplerFilter::LINEAR;
                lSamplingInfo.mMagnification = LTSE::Core::eSamplerFilter::LINEAR;
                lSamplingInfo.mWrapping      = LTSE::Core::eSamplerWrapping::CLAMP_TO_EDGE;

                return New<LTSE::Cuda::TextureSampler2D>( lTexture, lSamplingInfo );
            },
            []( Core::TextureData2D &aTexture, Core::TextureSampler2D &aSampler )
            {
                TextureData2D::sCreateInfo lCreateInfo{};

                LTSE::Cuda::sTextureCreateInfo lTextureCreateInfo{};
                lTextureCreateInfo.mFilterMode            = LTSE::Core::eSamplerFilter::LINEAR;
                lTextureCreateInfo.mWrappingMode          = LTSE::Core::eSamplerWrapping::CLAMP_TO_EDGE;
                lTextureCreateInfo.mNormalizedCoordinates = true;

                switch( aTexture.mSpec.mFormat )
                {
                case LTSE::Core::eColorFormat::R32_FLOAT:
                    lTextureCreateInfo.mNormalizedValues = false;
                    break;
                default:
                    lTextureCreateInfo.mNormalizedValues = true;
                    break;
                }

                Ref<LTSE::Cuda::Texture2D> lTexture = New<LTSE::Cuda::Texture2D>( lTextureCreateInfo, aTexture.GetImageData() );

                return New<LTSE::Cuda::TextureSampler2D>( lTexture, aSampler.mSamplingSpec );
            } );
    }
}; // namespace LTSE::Core