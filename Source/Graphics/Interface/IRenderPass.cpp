#include "IRenderPass.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    IRenderPass::IRenderPass( Ref<IGraphicContext> aGraphicContext, uint32_t aSampleCount )
        : mGraphicContext{ aGraphicContext }
        , mSampleCount{ aSampleCount }
    {
    }
} // namespace SE::Graphics