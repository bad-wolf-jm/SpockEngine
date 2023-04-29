#include "IRenderPass.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    IRenderPass::IRenderPass( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
        : IRenderTarget( aGraphicContext, sRenderTargetDescription{} )
        , mViewportClient{ aWindow }
    {
    }
} // namespace SE::Graphics