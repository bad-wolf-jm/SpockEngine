#include "ISwapChain.h"

#include "Core/Memory.h"
#include <stdexcept>

namespace SE::Graphics
{
    ISwapChain::ISwapChain( ref_t<IGraphicContext> aGraphicContext, ref_t<IWindow> aWindow )
        : IRenderTarget( aGraphicContext, sRenderTargetDescription{} )
        , mViewportClient{ aWindow }
    {
    }
} // namespace SE::Graphics