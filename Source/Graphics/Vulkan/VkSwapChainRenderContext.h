#pragma once

#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/IRenderContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "VkBaseRenderContext.h"
#include "VkRenderTarget.h"

namespace SE::Graphics
{
    class VkSwapChainRenderContext : public VkBaseRenderContext
    {
      public:
        VkSwapChainRenderContext() = default;
        VkSwapChainRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget );

        ~VkSwapChainRenderContext() = default;

        bool BeginRender();
    };

} // namespace SE::Graphics