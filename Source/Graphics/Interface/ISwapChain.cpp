#include "ISwapChain.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    ISwapChain::ISwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow )
        : IRenderTarget( aGraphicContext, sRenderTargetDescription{} )
        , mViewportClient{ aWindow }
    {
    }
} // namespace SE::Graphics