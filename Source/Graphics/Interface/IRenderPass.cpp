#include "IRenderPass.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    IRenderPass::IRenderPass( ref_t<IGraphicContext> aGraphicContext, uint32_t aSampleCount )
        : mGraphicContext{ aGraphicContext }
        , mSampleCount{ aSampleCount }
    {
    }
} // namespace SE::Graphics