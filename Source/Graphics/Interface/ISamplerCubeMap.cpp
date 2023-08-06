#include "ISamplerCubeMap.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    /** @brief */
    ISamplerCubeMap::ISamplerCubeMap( ref_t<IGraphicContext> aGraphicContext, ref_t<ITextureCubeMap> aTextureData,
                                      sTextureSamplingInfo const &aSamplingSpec )
        : mGraphicContext( aGraphicContext )
        , mTextureData{ aTextureData }
    {
        mSpec    = aSamplingSpec;
        mTexture = aTextureData;
    }
} // namespace SE::Graphics
