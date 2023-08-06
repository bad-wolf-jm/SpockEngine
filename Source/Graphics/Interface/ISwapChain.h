#pragma once

#include "Core/Memory.h"
#include "IGraphicContext.h"
#include "IRenderTarget.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class ISwapChain : public IRenderTarget
    {
      public:
        ISwapChain( ref_t<IGraphicContext> aGraphicContext, ref_t<IWindow> aWindow );
        ~ISwapChain() = default;

      protected:
        ref_t<IWindow> mViewportClient = nullptr;
    };

} // namespace SE::Graphics