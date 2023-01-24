#include "ISamplerCubeMap.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    /** @brief */
    ISamplerCubeMap::ISamplerCubeMap( Ref<IGraphicContext> aGraphicContext, Ref<ITextureCubeMap> aTextureData,
                                      sTextureSamplingInfo const &aSamplingSpec )
        : mGraphicContext( aGraphicContext )
        , mTextureData{ aTextureData }
    {
        mSpec    = aSamplingSpec;
        mTexture = aTextureData;
    }
} // namespace SE::Graphics
