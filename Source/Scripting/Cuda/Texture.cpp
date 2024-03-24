#include "Texture.h"

#include "Core/CUDA/Texture/Texture2D.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Definitions.h"

#include "TensorOps/Scope.h"

#include "Core/Logging.h"

#include "Scripting/Core/Texture.h"

namespace SE::Core
{
    using namespace sol;

    Cuda::texture_create_info_t parse_cuda_create_info( sol::table table )
    {
        Cuda::texture_create_info_t lCreateInfo{};
        // lCreateInfo.mFilterMode            = table["filter_mode"].valid() ? table["filter_mode"] : eSamplerFilter::LINEAR;
        // lCreateInfo.mWrappingMode          = table["wrapping"].valid() ? table["wrapping"] : eSamplerWrapping::CLAMP_TO_BORDER;
        // lCreateInfo.mFormat                = table["color_format"].valid() ? table["color_format"] : eColorFormat::UNDEFINED;
        // lCreateInfo.mWidth                 = table["width"].valid() ? table["width"] : 0;
        // lCreateInfo.mHeight                = table["height"].valid() ? table["height"] : 0;
        // lCreateInfo.mNormalizedCoordinates = table["normalized_coordinates"].valid() ? table["normalized_coordinates"] : false;
        // lCreateInfo.mNormalizedValues      = table["normalized_values"].valid() ? table["normalized_values"] : false;

        return lCreateInfo;
    }

    void require_cuda_texture( sol::table &scriptingState )
    {
        auto textureData2DType = scriptingState.new_usertype<Cuda::texture2d_t>( "Texture2D" );

        // clang-format off
        textureData2DType[call_constructor] = factories(
            []( sol::table createInfo, sol::table imageData ) {
                return New<Cuda::texture2d_t>( parse_cuda_create_info( createInfo ), parse_image_data( imageData ) );
            },
            []( sol::table createInfo, vector_t<uint8_t> imageData ) {
                return New<Cuda::texture2d_t>( parse_cuda_create_info( createInfo ), imageData );
            }
        );
        // clang-format on

        auto textureSampler2DType = scriptingState.new_usertype<Cuda::texture_sampler2d_t>( "TextureSampler2D" );

        // clang-format off
        textureSampler2DType[call_constructor] = factories( 
                []( ref_t<Cuda::texture2d_t> &texture, sol::table createInfo )
                { 
                    return Cuda::texture_sampler2d_t( texture, parse_sampler_info( createInfo ) ); 
                }
            );
        // clang-format on
    }
}; // namespace SE::Core