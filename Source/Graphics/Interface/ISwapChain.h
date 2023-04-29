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
        ISwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow );
        ~ISwapChain() = default;

      protected:
        Ref<IWindow> mViewportClient = nullptr;
    };

} // namespace SE::Graphics