#include "ISampler2D.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"

namespace SE::Graphics
{
    /** @brief */
    ISampler2D::ISampler2D( Ref<IGraphicContext> aGraphicContext, Ref<ITexture2D> aTextureData,
                            sTextureSamplingInfo const &aSamplingSpec )
        : mGraphicContext( aGraphicContext )
        , mTextureData{ aTextureData }
    {
        mSpec    = aSamplingSpec;
        mTexture = aTextureData;
    }
} // namespace SE::Graphics
