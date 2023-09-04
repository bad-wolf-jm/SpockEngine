#include "Texture.h"

#include "Core/Memory.h"
#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/Texture2D.h"

#include "TensorOps/Scope.h"

#include "Core/Logging.h"

#include "Scripting/Core/Texture.h"

namespace SE::Core
{
    using namespace sol;

    Cuda::sTextureCreateInfo ParseCudaCreateInfo( sol::table aTable )
    {
        Cuda::sTextureCreateInfo lCreateInfo{};
        // lCreateInfo.mFilterMode            = aTable["filter_mode"].valid() ? aTable["filter_mode"] : eSamplerFilter::LINEAR;
        // lCreateInfo.mWrappingMode          = aTable["wrapping"].valid() ? aTable["wrapping"] : eSamplerWrapping::CLAMP_TO_BORDER;
        // lCreateInfo.mFormat                = aTable["color_format"].valid() ? aTable["color_format"] : eColorFormat::UNDEFINED;
        // lCreateInfo.mWidth                 = aTable["width"].valid() ? aTable["width"] : 0;
        // lCreateInfo.mHeight                = aTable["height"].valid() ? aTable["height"] : 0;
        // lCreateInfo.mNormalizedCoordinates = aTable["normalized_coordinates"].valid() ? aTable["normalized_coordinates"] : false;
        // lCreateInfo.mNormalizedValues      = aTable["normalized_values"].valid() ? aTable["normalized_values"] : false;

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
            []( sol::table aCreateInfo, vector_t<uint8_t> aImageData ) {
                return New<Cuda::Texture2D>( ParseCudaCreateInfo( aCreateInfo ), aImageData );
            }
        );
        // clang-format on

        auto lTextureSampler2DType = aScriptingState.new_usertype<Cuda::TextureSampler2D>( "TextureSampler2D" );

        lTextureSampler2DType[call_constructor] =
            factories( []( ref_t<Cuda::Texture2D> &aTexture, sol::table aCreateInfo ) { return Cuda::TextureSampler2D( aTexture, ParseSamplerInfo( aCreateInfo ) ); } );
    }
}; // namespace SE::Core